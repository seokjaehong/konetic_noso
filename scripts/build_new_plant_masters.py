"""
영동 / 여수 발전본부 master CSV 빌드

공통 전처리:
  - 배출: 농도×유량×1440/1e6 → kg/일, 호기 합산
  - 발전실적: 월별 가중평균(이용률/열효율), 균등배분(발전량)
  - 연료소비: 월별 균등배분 → 일별
  - 기상: 영동=koenergy XLS 우선+Open-Meteo 보완 / 여수=Open-Meteo
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import requests

RAW_DIR  = _ROOT / 'data' / 'raw'
PROC_DIR = _ROOT / 'data' / 'processed'

PLANT_COORDS = {
    '영동': {'lat': 37.1756, 'lon': 129.3610},
    '여수': {'lat': 34.7604, 'lon': 127.7442},
}

PLANT_CAPACITY_MW = {
    '영동': 400,   # 200 MW × 2
    '여수': 680,   # 340 MW × 2
}

# ────────────────────────────────────────────────────────
# 공통 헬퍼
# ────────────────────────────────────────────────────────

def fetch_open_meteo(plant: str, start='2020-07-01', end='2026-04-30') -> pd.DataFrame:
    lat = PLANT_COORDS[plant]['lat']
    lon = PLANT_COORDS[plant]['lon']
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min"
        ",relative_humidity_2m_mean,wind_speed_10m_mean,wind_speed_10m_max"
        ",wind_direction_10m_dominant,precipitation_sum,shortwave_radiation_sum"
        "&wind_speed_unit=ms"   # 기본값 km/h → m/s 명시
        "&timezone=Asia%2FSeoul"
    )
    print(f"  Open-Meteo 수집: {plant} ({lat},{lon})...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()['daily']
    df = pd.DataFrame(data)
    df.rename(columns={
        'time': 'date',
        'temperature_2m_mean': 'temp_mean',
        'temperature_2m_max': 'temp_max',
        'temperature_2m_min': 'temp_min',
        'relative_humidity_2m_mean': 'humidity_mean',
        'wind_speed_10m_mean': 'wind_speed_mean',
        'wind_speed_10m_max': 'wind_speed_max',
        'wind_direction_10m_dominant': 'wind_dir',
        'precipitation_sum': 'precipitation',
        'shortwave_radiation_sum': 'solar_radiation',
    }, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    wind_rad = np.deg2rad(df['wind_dir'].fillna(0))
    df['wind_sin'] = np.sin(wind_rad)
    df['wind_cos'] = np.cos(wind_rad)
    df['stagnation_idx'] = df['humidity_mean'] / df['wind_speed_mean'].clip(lower=0.1)
    df['pressure_mean'] = np.nan
    save_path = RAW_DIR / f'weather_{plant}_{start}_{end}.csv'
    df.to_csv(save_path, index=False)
    print(f"  저장: {save_path} ({len(df)}행)")
    return df


def load_open_meteo(plant: str, start='2020-07-01', end='2026-04-30') -> pd.DataFrame:
    path = RAW_DIR / f'weather_{plant}_{start}_{end}.csv'
    if path.exists():
        df = pd.read_csv(path, parse_dates=['date'])
        print(f"  Open-Meteo 캐시 로드: {plant} ({len(df)}행)")
        return df
    return fetch_open_meteo(plant, start, end)


def load_koenergy_weather(plant: str) -> pd.DataFrame:
    df_w = pd.read_excel(RAW_DIR / '한국남동발전_기상정보(일평균).xls', header=None)
    df_w.columns = df_w.iloc[0]
    df_w = df_w.iloc[1:].reset_index(drop=True)
    sub = df_w[df_w['사업소'] == plant].copy()
    sub['date'] = pd.to_datetime(sub['일자'].astype(str), format='%Y%m%d', errors='coerce')
    sub = sub.dropna(subset=['date'])
    for col in ['온도', '습도', '풍향', '풍속', '강수량']:
        sub[col] = pd.to_numeric(sub[col], errors='coerce')
    wx = sub.groupby('date').agg(
        temp_mean=('온도', 'mean'),
        humidity_mean=('습도', 'mean'),
        wind_dir=('풍향', 'mean'),
        wind_speed_mean=('풍속', 'mean'),
        precipitation=('강수량', 'mean'),
    ).reset_index()
    wind_rad = np.deg2rad(wx['wind_dir'].fillna(0))
    wx['wind_sin'] = np.sin(wind_rad)
    wx['wind_cos'] = np.cos(wind_rad)
    wx['stagnation_idx'] = wx['humidity_mean'] / wx['wind_speed_mean'].clip(lower=0.1)
    return wx


def load_emissions(plant: str) -> pd.DataFrame:
    em = pd.read_excel(RAW_DIR / '한국남동발전_대기오염물질배출농도(일평균).xls', header=None)
    em.columns = em.iloc[0]
    em = em.iloc[1:].reset_index(drop=True)
    sub = em[em['사업소'] == plant].copy()
    sub['date'] = pd.to_datetime(sub['일자'].astype(str), format='%Y%m%d', errors='coerce')
    sub = sub.dropna(subset=['date'])
    for col in ['SOX', 'NOX', '먼지', '유량']:
        sub[col] = pd.to_numeric(sub[col], errors='coerce')
    sub['SOx_kg'] = sub['SOX'] * sub['유량'] * 1440 / 1e6
    sub['NOx_kg'] = sub['NOX'] * sub['유량'] * 1440 / 1e6
    sub['dust_kg'] = sub['먼지'] * sub['유량'] * 1440 / 1e6
    daily = sub.groupby('date').agg(
        SOx=('SOx_kg', 'sum'),
        NOx=('NOx_kg', 'sum'),
        먼지=('dust_kg', 'sum'),
        n_units=('호기', 'count'),
    ).reset_index()
    return daily


def load_gen_xls() -> pd.DataFrame:
    df = pd.read_excel(RAW_DIR / '한국남동발전_발전실적.xls', header=0)
    df.columns = ['사업소', '호기', '일자', '용량_MW', '발전량_MWh', '열효율', '이용률', '발전원']
    for col in ['발전량_MWh', '열효율', '이용률', '용량_MW']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['일자_str'] = df['일자'].apply(lambda x: str(int(x)) if pd.notna(x) else '')
    df = df[df['일자_str'].str.len() == 6].copy()
    df['year']  = df['일자_str'].str[:4].astype(int)
    df['month'] = df['일자_str'].str[4:].astype(int)
    return df


def build_gen_monthly(df_xls: pd.DataFrame, plant: str) -> pd.DataFrame:
    sub = df_xls[df_xls['사업소'] == plant].dropna(subset=['발전량_MWh']).copy()

    monthly_gen = sub.groupby(['year', 'month'])['발전량_MWh'].sum().reset_index(name='monthly_gen_mwh')

    def weighted_util(g):
        mask = g['발전량_MWh'] > 0
        if mask.sum() == 0:
            return np.nan
        return (g.loc[mask, '발전량_MWh'] * g.loc[mask, '이용률']).sum() / g.loc[mask, '발전량_MWh'].sum()

    def weighted_eff(g):
        mask = (g['발전량_MWh'] > 0) & (g['열효율'] > 1) & (g['열효율'] <= 100)
        if mask.sum() == 0:
            return np.nan
        return (g.loc[mask, '발전량_MWh'] * g.loc[mask, '열효율']).sum() / g.loc[mask, '발전량_MWh'].sum()

    monthly_util = sub.groupby(['year', 'month']).apply(weighted_util).reset_index(name='utilization')
    monthly_eff  = sub.groupby(['year', 'month']).apply(weighted_eff).reset_index(name='heat_efficiency')

    monthly = monthly_gen.merge(monthly_util, on=['year', 'month']).merge(monthly_eff, on=['year', 'month'])
    return monthly


def monthly_to_daily_df(monthly: pd.DataFrame, date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """월별 발전량/이용률/열효율 → 일별 DataFrame"""
    base = pd.DataFrame({'date': date_range})
    base['year']  = base['date'].dt.year
    base['month'] = base['date'].dt.month
    base['days']  = base.apply(
        lambda r: pd.Period(f'{r.year}-{r.month:02d}').days_in_month, axis=1
    )
    merged = base.merge(monthly, on=['year', 'month'], how='left')
    merged['gen_mwh'] = (merged['monthly_gen_mwh'] / merged['days']).ffill().bfill()
    merged['utilization']    = merged['utilization'].ffill().bfill()
    merged['heat_efficiency'] = merged['heat_efficiency'].ffill().bfill()
    return merged[['date', 'gen_mwh', 'utilization', 'heat_efficiency']]


def load_fuel(plant: str, fuel_col_map: dict) -> pd.DataFrame:
    df = pd.read_excel(RAW_DIR / '한국남동발전_연료소비실적.xls', header=0)
    df.columns = ['사업소', '호기', '일자', '유연탄', '무연탄', '계_석탄', '유류', 'LNG', '고형연료', '우드펠릿']
    sub = df[df['사업소'] == plant].copy()
    sub['ym'] = pd.to_datetime(sub['일자'].astype(str).str.zfill(6), format='%Y%m', errors='coerce')
    sub = sub.dropna(subset=['ym'])
    for col in fuel_col_map.keys():
        sub[col] = pd.to_numeric(sub[col], errors='coerce').fillna(0)
    monthly = sub.groupby('ym')[list(fuel_col_map.keys())].sum().reset_index()
    monthly['year']  = monthly['ym'].dt.year
    monthly['month'] = monthly['ym'].dt.month
    return monthly


def fuel_monthly_to_daily(monthly_fuel: pd.DataFrame, fuel_cols: list,
                           date_range: pd.DatetimeIndex) -> pd.DataFrame:
    base = pd.DataFrame({'date': date_range})
    base['year']  = base['date'].dt.year
    base['month'] = base['date'].dt.month
    base['days']  = base.apply(
        lambda r: pd.Period(f'{r.year}-{r.month:02d}').days_in_month, axis=1
    )
    merged = base.merge(monthly_fuel[['year', 'month'] + fuel_cols], on=['year', 'month'], how='left')
    for col in fuel_cols:
        merged[col] = (merged[col] / merged['days']).ffill().bfill()
    return merged[['date'] + fuel_cols]


def add_time_features(master: pd.DataFrame) -> pd.DataFrame:
    master['month']     = master['date'].dt.month
    master['dayofweek'] = master['date'].dt.dayofweek
    master['year']      = master['date'].dt.year
    master['season']    = master['month'].map({
        12:'winter', 1:'winter', 2:'winter',
        3:'spring',  4:'spring', 5:'spring',
        6:'summer',  7:'summer', 8:'summer',
        9:'autumn', 10:'autumn', 11:'autumn',
    })
    master['seasonal_mgmt'] = master['month'].isin([12,1,2,3]).astype(int)
    return master


def add_lag_features(master: pd.DataFrame, targets: list) -> pd.DataFrame:
    master = master.sort_values('date').reset_index(drop=True)
    for t in targets:
        if t in master.columns:
            master[f'{t}_lag1'] = master[t].shift(1)
            master[f'{t}_lag7'] = master[t].shift(7)
    return master


# ────────────────────────────────────────────────────────
# 영동 master 빌드
# ────────────────────────────────────────────────────────

def _load_api_gen(plant: str) -> pd.DataFrame:
    """영동/여수 API 일별 발전량 로딩 (thermal_gen_..._yeongdong_yeosu.csv)"""
    api_path = RAW_DIR / 'thermal_gen_20220101_20260509_yeongdong_yeosu.csv'
    if not api_path.exists():
        return pd.DataFrame(columns=['date', 'api_gen_mwh'])
    df = pd.read_csv(api_path, parse_dates=['date'])
    df = df[df['plant_name'] == plant][['date', 'daily_gen_mwh']].copy()
    df = df.rename(columns={'daily_gen_mwh': 'api_gen_mwh'})
    # 0값은 실제 결측으로 간주 (발전소 완전 정지는 gen_mwh로 보완)
    df.loc[df['api_gen_mwh'] <= 0, 'api_gen_mwh'] = np.nan
    return df.reset_index(drop=True)


def build_yeongdong(df_xls: pd.DataFrame):
    print('\n' + '='*65)
    print('  영동 발전본부 master 빌드')
    print('='*65)

    em = load_emissions('영동')
    date_range = pd.date_range(em['date'].min(), em['date'].max(), freq='D')

    # 발전 월별 → 일별
    gen_monthly = build_gen_monthly(df_xls, '영동')
    gen_daily   = monthly_to_daily_df(gen_monthly, date_range)

    # 기상: 전체 date_range 기준 → koenergy 우선, 결측은 Open-Meteo 보완
    wx_k  = load_koenergy_weather('영동')
    wx_om = load_open_meteo('영동')
    print(f'  koenergy 기상: {len(wx_k)}행')

    wx_base = pd.DataFrame({'date': date_range})
    wx = wx_base.merge(wx_k, on='date', how='left')
    wx = wx.merge(wx_om[['date','temp_mean','humidity_mean','wind_speed_mean',
                           'wind_speed_max','wind_sin','wind_cos','stagnation_idx',
                           'solar_radiation','precipitation']],
                  on='date', how='left', suffixes=('', '_om'))
    for field in ['temp_mean','humidity_mean','wind_speed_mean','precipitation']:
        om_col = field + '_om'
        if om_col in wx.columns:
            wx[field] = wx[field].fillna(wx[om_col])
    for field in ['wind_sin','wind_cos','stagnation_idx','solar_radiation','wind_speed_max']:
        om_col = field + '_om'
        if field not in wx.columns and om_col in wx.columns:
            wx[field] = wx[om_col]
        elif om_col in wx.columns:
            wx[field] = wx[field].fillna(wx[om_col])
    # wind_sin/cos 재계산 (koenergy wind_dir 기준)
    if 'wind_dir' in wx.columns:
        wind_rad = np.deg2rad(wx['wind_dir'].fillna(0))
        wx['wind_sin'] = wx['wind_sin'].fillna(np.sin(wind_rad))
        wx['wind_cos'] = wx['wind_cos'].fillna(np.cos(wind_rad))
    if 'stagnation_idx' not in wx.columns or wx['stagnation_idx'].isna().any():
        wx['stagnation_idx'] = wx['stagnation_idx'].fillna(
            wx['humidity_mean'] / wx['wind_speed_mean'].clip(lower=0.1))
    wx['pressure_mean'] = np.nan  # koenergy 기압 100% NaN, Open-Meteo 미제공

    # 연료: 유연탄+무연탄+우드펠릿+유류
    fuel_cols = ['유연탄', '무연탄', '유류', '우드펠릿']
    fuel_monthly = load_fuel('영동', {c: c for c in fuel_cols})
    fuel_daily   = fuel_monthly_to_daily(fuel_monthly, fuel_cols, date_range)

    # API 일별 발전량
    api_gen = _load_api_gen('영동')
    print(f'  API 발전량: {api_gen["api_gen_mwh"].notna().sum()}일 유효')

    # 통합
    master = pd.DataFrame({'date': date_range})
    master = master.merge(em, on='date', how='left')
    master = master.merge(gen_daily, on='date', how='left')
    wx_cols = ['date','temp_mean','humidity_mean','wind_speed_mean',
               'wind_speed_max','wind_sin','wind_cos','precipitation',
               'stagnation_idx','solar_radiation']
    master = master.merge(wx[[c for c in wx_cols if c in wx.columns]], on='date', how='left')
    master = master.merge(fuel_daily, on='date', how='left')
    master = master.merge(api_gen, on='date', how='left')

    # 파생: API 실측 우선, 결측은 월별 균등배분 보완
    master['gen_mwh_combined'] = master['api_gen_mwh'].fillna(master['gen_mwh'])
    total_fuel = (master['유연탄'].fillna(0) + master['무연탄'].fillna(0)
                  + master['우드펠릿'].fillna(0) + master['유류'].fillna(0)).clip(lower=0.001)
    master['coal_ratio']     = (master['유연탄'].fillna(0) + master['무연탄'].fillna(0)) / total_fuel
    master['biomass_ratio']  = master['우드펠릿'].fillna(0) / total_fuel
    master['oil_ratio']      = master['유류'].fillna(0) / total_fuel
    master['utilization']    = master['utilization'].fillna(0)
    master['renewable_ratio'] = 0.0
    master['solar_mwh']      = 0.0
    master['wind_mwh']       = 0.0
    master['is_coal']        = 1
    master['plant']          = '영동'

    master = add_time_features(master)
    master = add_lag_features(master, ['SOx', 'NOx', '먼지'])

    out = PROC_DIR / 'master_영동.csv'
    master.to_csv(out, index=False)
    api_filled = master['api_gen_mwh'].notna().sum()
    print(f'  [OK] 저장: {out} ({len(master)}행 × {len(master.columns)}컬럼)')
    print(f'  기간: {master.date.min().date()} ~ {master.date.max().date()}')
    print(f'  gen_mwh_combined: API실측 {api_filled}일 / 균등배분 {len(master)-api_filled}일')
    print(f'  NOx 평균: {master.NOx.mean():.1f} kg/일')
    print(f'  gen_mwh_combined 평균: {master.gen_mwh_combined.mean():.0f} MWh/일 (vs 구 gen_mwh {master.gen_mwh.mean():.0f})')
    return master


# ────────────────────────────────────────────────────────
# 여수 master 빌드
# ────────────────────────────────────────────────────────

def build_yeosu(df_xls: pd.DataFrame):
    print('\n' + '='*65)
    print('  여수 발전본부 master 빌드')
    print('='*65)

    em = load_emissions('여수')
    date_range = pd.date_range(em['date'].min(), em['date'].max(), freq='D')

    # 발전 월별 → 일별
    gen_monthly = build_gen_monthly(df_xls, '여수')
    gen_daily   = monthly_to_daily_df(gen_monthly, date_range)

    # 기상: Open-Meteo (koenergy에 여수 없음)
    wx = load_open_meteo('여수')
    wx['pressure_mean'] = np.nan

    # 연료: 유연탄+유류+우드펠릿
    fuel_cols = ['유연탄', '유류', '우드펠릿']
    fuel_monthly = load_fuel('여수', {c: c for c in fuel_cols})
    fuel_daily   = fuel_monthly_to_daily(fuel_monthly, fuel_cols, date_range)

    # API 일별 발전량
    api_gen = _load_api_gen('여수')
    print(f'  API 발전량: {api_gen["api_gen_mwh"].notna().sum()}일 유효')

    # 통합
    master = pd.DataFrame({'date': date_range})
    master = master.merge(em, on='date', how='left')
    master = master.merge(gen_daily, on='date', how='left')
    wx_cols = ['date','temp_mean','humidity_mean','wind_speed_mean',
               'wind_speed_max','wind_sin','wind_cos','precipitation',
               'stagnation_idx','solar_radiation']
    master = master.merge(wx[[c for c in wx_cols if c in wx.columns]], on='date', how='left')
    master = master.merge(fuel_daily, on='date', how='left')
    master = master.merge(api_gen, on='date', how='left')

    # 파생: API 실측 우선, 결측은 월별 균등배분 보완
    master['gen_mwh_combined'] = master['api_gen_mwh'].fillna(master['gen_mwh'])
    total_fuel = (master['유연탄'].fillna(0) + master['유류'].fillna(0)
                  + master['우드펠릿'].fillna(0)).clip(lower=0.001)
    master['coal_ratio']   = master['유연탄'].fillna(0) / total_fuel
    master['oil_ratio']    = master['유류'].fillna(0) / total_fuel
    master['biomass_ratio'] = master['우드펠릿'].fillna(0) / total_fuel
    master['utilization']  = master['utilization'].fillna(0)
    master['renewable_ratio'] = 0.0
    master['solar_mwh']    = 0.0
    master['wind_mwh']     = 0.0
    master['is_coal']      = 1
    master['plant']        = '여수'

    master = add_time_features(master)
    master = add_lag_features(master, ['SOx', 'NOx', '먼지'])

    out = PROC_DIR / 'master_여수.csv'
    master.to_csv(out, index=False)
    api_filled = master['api_gen_mwh'].notna().sum()
    print(f'  [OK] 저장: {out} ({len(master)}행 × {len(master.columns)}컬럼)')
    print(f'  기간: {master.date.min().date()} ~ {master.date.max().date()}')
    print(f'  gen_mwh_combined: API실측 {api_filled}일 / 균등배분 {len(master)-api_filled}일')
    print(f'  NOx 평균: {master.NOx.mean():.1f} kg/일')
    print(f'  gen_mwh_combined 평균: {master.gen_mwh_combined.mean():.0f} MWh/일 (vs 구 gen_mwh {master.gen_mwh.mean():.0f})')
    return master


if __name__ == '__main__':
    print('발전실적.xls 로딩...')
    df_xls = load_gen_xls()

    build_yeongdong(df_xls)
    build_yeosu(df_xls)

    print('\n=== 전체 완료 ===')
    for plant in ['영동', '여수']:
        df = pd.read_csv(PROC_DIR / f'master_{plant}.csv')
        print(f'{plant}: {len(df)}행, NOx mean={df.NOx.mean():.1f} kg/일, '
              f'utilization mean={df.utilization.mean():.1f}%')
