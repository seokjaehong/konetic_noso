"""
신재생 발전량 연결 스크립트 (Step 0)
- every/태양광·풍력·해양소수력 xlsx → 사업소별 일별 MWh 집계
- master_삼천포.csv, master_영흥.csv의 solar_mwh·wind_mwh 업데이트
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))

import pandas as pd
import glob
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
EVERY_DIR = RAW_DIR / "every"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# ── 발전구분 필터 ──────────────────────────────────────────────
PLANT_FILTERS = {
    "삼천포": {
        "solar":  ["삼천포태양광", "삼천포태양광#5", "삼천포태양광#6"],
        "wind":   ["삼천포풍력"],
        "hydro":  ["삼천포해양소수력"],
    },
    "영흥": {
        "solar":  ["영흥태양광", "영흥태양광 #3", "영흥태양광#5"],
        "wind":   ["영흥풍력"],
        "hydro":  ["영흥해양소수력"],
    },
}


def load_category(prefix: str) -> pd.DataFrame:
    """every/{prefix}_YYYYMM.xlsx 전체 로드 → 발전구분·일자·총량 반환"""
    files = sorted(glob.glob(str(EVERY_DIR / f"{prefix}_*.xlsx")))
    files = [f for f in files if "after_search" not in f]
    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f)
            dfs.append(df)
        except Exception as e:
            print(f"  [WARN] {f}: {e}")
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["일자"]).dt.normalize()

    # 총량 컬럼 찾기 (이름이 파일마다 다름: '총량(KW)', '총량(MW)')
    total_col = [c for c in combined.columns if "총량" in c]
    if not total_col:
        print(f"  [WARN] {prefix}: 총량 컬럼 없음")
        return pd.DataFrame()
    total_col = total_col[0]

    combined["total_kwh"] = pd.to_numeric(combined[total_col], errors="coerce")
    return combined[["발전구분", "date", "total_kwh"]]


def aggregate_plant(df: pd.DataFrame, plant_names: list[str]) -> pd.Series:
    """특정 발전구분 필터 후 일별 MWh 집계 (KWh / 1000)"""
    mask = df["발전구분"].isin(plant_names)
    filtered = df[mask]
    daily_kwh = filtered.groupby("date")["total_kwh"].sum()
    daily_mwh = daily_kwh / 1000
    return daily_mwh


# ── 파일 로딩 ─────────────────────────────────────────────────
print("=== 신재생 발전량 로딩 ===")
df_solar = load_category("태양광")
df_wind  = load_category("풍력")
df_hydro = load_category("해양소수력")

print(f"  태양광: {len(df_solar)}행, 기간: {df_solar['date'].min().date()} ~ {df_solar['date'].max().date()}")
print(f"  풍력:   {len(df_wind)}행, 기간: {df_wind['date'].min().date()} ~ {df_wind['date'].max().date()}")
print(f"  소수력: {len(df_hydro)}행, 기간: {df_hydro['date'].min().date()} ~ {df_hydro['date'].max().date()}")

# ── 사업소별 집계 및 마스터 업데이트 ─────────────────────────
for plant_key, filters in PLANT_FILTERS.items():
    master_path = PROCESSED_DIR / f"master_{plant_key}.csv"
    if not master_path.exists():
        print(f"\n[SKIP] {master_path} 없음")
        continue

    print(f"\n=== {plant_key} 마스터 업데이트 ===")
    master = pd.read_csv(master_path, parse_dates=["date"])
    n_orig = len(master)

    # 일별 MWh 집계
    solar_daily = aggregate_plant(df_solar, filters["solar"])
    wind_daily  = aggregate_plant(df_wind,  filters["wind"])
    hydro_daily = aggregate_plant(df_hydro, filters["hydro"])

    print(f"  태양광 일수: {len(solar_daily)}, 평균: {solar_daily.mean():.1f} MWh/day")
    print(f"  풍력 일수:   {len(wind_daily)}, 평균: {wind_daily.mean():.1f} MWh/day")
    print(f"  소수력 일수: {len(hydro_daily)}, 평균: {hydro_daily.mean():.1f} MWh/day")

    # 마스터에 병합
    master = master.set_index("date")
    master["solar_mwh"]  = solar_daily.reindex(master.index).fillna(0)
    master["wind_mwh"]   = wind_daily.reindex(master.index).fillna(0)
    master["hydro_mwh"]  = hydro_daily.reindex(master.index).fillna(0)

    # renewable_mwh·ratio 재계산
    master["renewable_mwh"] = master["solar_mwh"] + master["wind_mwh"] + master["hydro_mwh"]
    gen_base = master.get("gen_mwh_combined", master.get("api_gen_mwh", None))
    if gen_base is not None:
        master["renewable_ratio"] = (
            master["renewable_mwh"] /
            (gen_base.clip(lower=1) + master["renewable_mwh"])
        ).clip(0, 1)

    master = master.reset_index()
    assert len(master) == n_orig, "행 수 변경 이상"

    master.to_csv(master_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] 저장: {master_path}  ({len(master)} rows × {len(master.columns)} cols)")
    print(f"  solar_mwh 평균: {master['solar_mwh'].mean():.1f} MWh/day")
    print(f"  wind_mwh  평균: {master['wind_mwh'].mean():.1f} MWh/day")
    print(f"  renewable_ratio 평균: {master['renewable_ratio'].mean():.4f}")

print("\n[완료] 신재생 데이터 연결 완료")
