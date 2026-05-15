"""
데이터 수정 스크립트:
  이슈1: 영흥 gen_mwh / api_gen_mwh / utilization — 발전실적.xls 전 호기 합산으로 교체
  이슈2: 분당 heat_efficiency — 발전실적.xls CG호기 가중평균 열효율로 교체
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

PROCESSED_DIR = _ROOT / 'data' / 'processed'
RAW_DIR       = _ROOT / 'data' / 'raw'


def load_gen_xls() -> pd.DataFrame:
    df = pd.read_excel(RAW_DIR / '한국남동발전_발전실적.xls', sheet_name=0, header=0)
    df.columns = ['사업소', '호기', '일자', '용량_MW', '발전량_MWh', '열효율', '이용률', '발전원']
    for col in ['발전량_MWh', '열효율', '이용률', '용량_MW']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['일자_str'] = df['일자'].apply(lambda x: str(int(x)) if pd.notna(x) else '')
    df = df[df['일자_str'].str.len() == 6].copy()
    df['year']  = df['일자_str'].str[:4].astype(int)
    df['month'] = df['일자_str'].str[4:].astype(int)
    return df


def fix_yeongheung(df_xls: pd.DataFrame):
    """영흥 master CSV: gen_mwh / api_gen_mwh / utilization 재계산"""
    print('=== [이슈1] 영흥 gen_mwh / utilization 수정 ===')

    yh = df_xls[df_xls['사업소'] == '영흥'].dropna(subset=['발전량_MWh']).copy()

    # 월별 전 호기 합산 발전량
    monthly_gen = (yh
                   .groupby(['year', 'month'])['발전량_MWh']
                   .sum()
                   .reset_index(name='monthly_gen_mwh'))

    # 월별 가중평균 이용률 (발전량 > 0인 호기만)
    def weighted_util(g):
        mask = g['발전량_MWh'] > 0
        if mask.sum() == 0:
            return np.nan
        return (g.loc[mask, '발전량_MWh'] * g.loc[mask, '이용률']).sum() / g.loc[mask, '발전량_MWh'].sum()

    monthly_util = (yh
                    .groupby(['year', 'month'])
                    .apply(weighted_util)
                    .reset_index(name='monthly_util'))

    monthly = monthly_gen.merge(monthly_util, on=['year', 'month'])

    # master CSV 로딩
    path = PROCESSED_DIR / 'master_영흥.csv'
    master = pd.read_csv(path, parse_dates=['date'])
    master['year']  = master['date'].dt.year
    master['month'] = master['date'].dt.month

    # 각 날짜에 해당 월의 일수로 나눠 일별 발전량 계산
    days_in_month = master[['year', 'month', 'date']].copy()
    days_in_month['days'] = days_in_month.apply(
        lambda r: pd.Period(f'{r.year}-{r.month:02d}').days_in_month, axis=1
    )
    master = master.merge(days_in_month[['date', 'days']], on='date', how='left')
    master = master.merge(monthly, on=['year', 'month'], how='left')

    # 일별 발전량 = 월 합산 / 일수 (forward fill for missing months)
    master['gen_mwh_new'] = master['monthly_gen_mwh'] / master['days']
    master['gen_mwh_new'] = master['gen_mwh_new'].ffill().bfill()
    master['util_new']    = master['monthly_util'].ffill().bfill()

    # 수정 전 vs 후 비교
    print(f'  gen_mwh 수정 전: mean={master["gen_mwh"].mean():.0f}, max={master["gen_mwh"].max():.0f}')
    print(f'  gen_mwh 수정 후: mean={master["gen_mwh_new"].mean():.0f}, max={master["gen_mwh_new"].max():.0f}')
    print(f'  utilization 수정 전: mean={master["utilization"].mean():.1f}%')
    print(f'  utilization 수정 후: mean={master["util_new"].mean():.1f}%')
    print(f'  NaN 수정 후 gen: {master["gen_mwh_new"].isna().sum()}')

    # 적용
    master['gen_mwh']     = master['gen_mwh_new']
    master['api_gen_mwh'] = master['gen_mwh_new']
    master['utilization'] = master['util_new']

    # 임시 컬럼 제거
    master = master.drop(columns=['year', 'month', 'days', 'monthly_gen_mwh', 'monthly_util',
                                   'gen_mwh_new', 'util_new'])

    master.to_csv(path, index=False)
    print(f'  [OK] 저장: {path}')
    return master


def fix_samcheonpo(df_xls: pd.DataFrame):
    """삼천포 master CSV: utilization / heat_efficiency 가중평균으로 교체"""
    print('\n=== [이슈3] 삼천포 utilization / heat_efficiency 수정 ===')

    sc = df_xls[df_xls['사업소'] == '삼천포'].dropna(subset=['발전량_MWh']).copy()

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

    monthly_util = (sc.groupby(['year', 'month']).apply(weighted_util).reset_index(name='monthly_util'))
    monthly_eff  = (sc.groupby(['year', 'month']).apply(weighted_eff).reset_index(name='monthly_eff'))
    monthly = monthly_util.merge(monthly_eff, on=['year', 'month'])

    path = PROCESSED_DIR / 'master_삼천포.csv'
    master = pd.read_csv(path, parse_dates=['date'])
    master['year']  = master['date'].dt.year
    master['month'] = master['date'].dt.month
    master = master.merge(monthly, on=['year', 'month'], how='left')
    master['monthly_util'] = master['monthly_util'].ffill().bfill()
    master['monthly_eff']  = master['monthly_eff'].ffill().bfill()

    print(f'  utilization 수정 전: mean={master["utilization"].mean():.1f}%')
    print(f'  utilization 수정 후: mean={master["monthly_util"].mean():.1f}%')
    print(f'  heat_efficiency 수정 전: mean={master["heat_efficiency"].mean():.4f}')
    print(f'  heat_efficiency 수정 후: mean={master["monthly_eff"].mean():.2f}%')

    master['utilization']    = master['monthly_util']
    master['heat_efficiency'] = master['monthly_eff']
    master = master.drop(columns=['year', 'month', 'monthly_util', 'monthly_eff'])
    master.to_csv(path, index=False)
    print(f'  [OK] 저장: {path}')
    return master


def fix_bundang(df_xls: pd.DataFrame):
    """분당 master CSV: heat_efficiency 재계산"""
    print('\n=== [이슈2] 분당 heat_efficiency 수정 ===')

    bd = df_xls[df_xls['사업소'] == '분당'].dropna(subset=['발전량_MWh']).copy()
    # CG 호기만 (열효율 > 0, 가스터빈)
    # 열효율 > 0이고 물리적으로 가능한 범위(1~50%)만 사용
    # 202501 데이터에 CG6=133%, CG7=459% 등 명백한 오류값 존재
    bd_cg = bd[(bd['열효율'] > 1) & (bd['열효율'] <= 50)].copy()

    def weighted_eff(g):
        if g['발전량_MWh'].sum() <= 0:
            return np.nan
        return (g['발전량_MWh'] * g['열효율']).sum() / g['발전량_MWh'].sum()

    monthly_eff = (bd_cg
                   .groupby(['year', 'month'])
                   .apply(weighted_eff)
                   .reset_index(name='heat_efficiency_monthly'))

    # master CSV 로딩
    path = PROCESSED_DIR / 'master_분당.csv'
    master = pd.read_csv(path, parse_dates=['date'])
    master['year']  = master['date'].dt.year
    master['month'] = master['date'].dt.month

    master = master.merge(monthly_eff, on=['year', 'month'], how='left')
    master['heat_efficiency_monthly'] = master['heat_efficiency_monthly'].ffill().bfill()

    print(f'  heat_efficiency 수정 전: mean={master["heat_efficiency"].mean():.4f}, '
          f'range=[{master["heat_efficiency"].min():.4f}, {master["heat_efficiency"].max():.4f}]')
    print(f'  heat_efficiency 수정 후: mean={master["heat_efficiency_monthly"].mean():.2f}%, '
          f'range=[{master["heat_efficiency_monthly"].min():.2f}, {master["heat_efficiency_monthly"].max():.2f}]')
    print(f'  NaN 수정 후: {master["heat_efficiency_monthly"].isna().sum()}')

    master['heat_efficiency'] = master['heat_efficiency_monthly']
    master = master.drop(columns=['year', 'month', 'heat_efficiency_monthly'])

    master.to_csv(path, index=False)
    print(f'  [OK] 저장: {path}')
    return master


def main():
    print('발전실적.xls 로딩...')
    df_xls = load_gen_xls()

    fix_yeongheung(df_xls)
    fix_bundang(df_xls)
    fix_samcheonpo(df_xls)

    print('\n=== 수정 완료 검증 ===')
    for plant in ['영흥', '분당', '삼천포']:
        df = pd.read_csv(PROCESSED_DIR / f'master_{plant}.csv')
        print(f'\n{plant}:')
        if 'gen_mwh' in df.columns:
            print(f'  gen_mwh: mean={df.gen_mwh.mean():.0f}')
        if 'api_gen_mwh' in df.columns:
            print(f'  api_gen_mwh: mean={df.api_gen_mwh.mean():.0f}, '
                  f'zero={( df.api_gen_mwh == 0).sum()}')
        if 'utilization' in df.columns:
            print(f'  utilization: mean={df.utilization.mean():.1f}%')
        if 'heat_efficiency' in df.columns:
            print(f'  heat_efficiency: mean={df.heat_efficiency.mean():.2f}, '
                  f'range=[{df.heat_efficiency.min():.2f}, {df.heat_efficiency.max():.2f}]')

    print('\n[완료] 두 파일 수정 완료. 다음 단계: 모델 재학습 후 최적화 재실행')


if __name__ == '__main__':
    main()
