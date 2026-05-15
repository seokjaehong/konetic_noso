"""
영흥 발전소 일별 발전량 데이터 수집 스크립트
기존 청크 캐시는 삼천포 전용이므로 영흥 전용 캐시 파일명을 사용해 신규 수집
"""
import sys
import os
import pandas as pd
import time
import requests
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import _get_api_key, _fetch_thermal_one_day, PLANT_CODES, RAW_DIR

# ── API 키 로딩 ─────────────────────────────────────────────────
key = _get_api_key()
if not key:
    print("[ERROR] API 키가 없습니다. .env 파일을 확인하세요.")
    sys.exit(1)
print(f"[OK] API 키 로딩 완료 (앞 10자): {key[:10]}...")

# ── 영흥 발전소 코드 ─────────────────────────────────────────────
TARGET_CODES = set(PLANT_CODES["영흥"])  # {"8641", "8642"}
CODE_TO_NAME = {c: "영흥" for c in TARGET_CODES}
print(f"[OK] 영흥 발전소 코드: {TARGET_CODES}")

# ── 수집 기간 ────────────────────────────────────────────────────
START = "2022-01-01"
END   = "2026-04-30"
date_range = pd.date_range(start=START, end=END, freq="D")
print(f"[OK] 수집 기간: {START} ~ {END} ({len(date_range)}일)")

# ── 영흥 전용 청크 캐시 디렉토리 (파일명에 _yh 접미사) ────────────
# 청크별로 yh 전용 캐시를 사용해 3개월씩 수집
chunk_months = 3
date_ranges_chunked = pd.date_range(start=START, end=END, freq=f"{chunk_months}MS")
end_ts = pd.Timestamp(END)

all_dfs = []
for i, chunk_start in enumerate(date_ranges_chunked):
    chunk_end = min(
        chunk_start + pd.DateOffset(months=chunk_months) - pd.Timedelta(days=1),
        end_ts
    )
    sd = chunk_start.strftime("%Y%m%d")
    ed = chunk_end.strftime("%Y%m%d")
    cache_path = RAW_DIR / f"thermal_gen_{sd}_{ed}_yh.csv"

    print(f"\n[청크 {i+1}/{len(date_ranges_chunked)}] {sd} ~ {ed}")

    # 캐시 있으면 재사용
    if cache_path.exists():
        df_chunk = pd.read_csv(cache_path, parse_dates=["date"])
        print(f"  [CACHE] {len(df_chunk)}행 로딩")
        all_dfs.append(df_chunk)
        continue

    # API 수집
    chunk_dates = pd.date_range(start=sd, end=ed, freq="D")
    records = []
    for j, dt in enumerate(chunk_dates):
        ds = dt.strftime("%Y%m%d")
        try:
            day_records = _fetch_thermal_one_day(key, ds, TARGET_CODES)
            records.extend(day_records)
        except Exception as e:
            print(f"    [WARN] {ds} 수집 실패: {e}")
        if (j + 1) % 30 == 0:
            print(f"    {j+1}/{len(chunk_dates)}일 완료...")
        time.sleep(0.1)

    if not records:
        print(f"  [WARN] 이 청크 데이터 없음")
        continue

    df_raw = pd.DataFrame(records)
    df_raw["date"] = pd.to_datetime(df_raw["dgenYmd"])
    df_raw["plant_name"] = df_raw["ippt"].map(CODE_TO_NAME)

    hour_cols = [f"qhorGen{k:02d}" for k in range(1, 25)]
    for col in hour_cols + ["qsum", "qavg"]:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    agg_dict = {"qsum": "sum", "qavg": "mean"}
    for col in hour_cols:
        if col in df_raw.columns:
            agg_dict[col] = "sum"

    df_chunk = (
        df_raw.groupby(["date", "plant_name"])
        .agg({k: v for k, v in agg_dict.items() if k in df_raw.columns})
        .reset_index()
    )
    df_chunk.rename(columns={"qsum": "daily_gen_kwh", "qavg": "avg_gen_kwh"}, inplace=True)
    if "daily_gen_kwh" in df_chunk.columns:
        df_chunk["daily_gen_mwh"] = df_chunk["daily_gen_kwh"] / 1000

    df_chunk.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] {len(df_chunk)}행 저장 → {cache_path.name}")
    all_dfs.append(df_chunk)

# ── 최종 통합 저장 ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("최종 통합 저장")
print("=" * 60)

if not all_dfs:
    print("[ERROR] 수집된 데이터 없음!")
    sys.exit(1)

final_df = (
    pd.concat(all_dfs, ignore_index=True)
    .drop_duplicates(subset=["date", "plant_name"])
    .sort_values("date")
    .reset_index(drop=True)
)

out_path = RAW_DIR / "thermal_gen_20220101_20260430_yeonghung.csv"
final_df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"\n저장 완료: {out_path}")
print(f"  총 행수: {len(final_df)}")
print(f"  날짜 범위: {final_df['date'].min().date()} ~ {final_df['date'].max().date()}")
print(f"  발전소: {final_df['plant_name'].unique().tolist()}")
print(f"\n컬럼 목록: {final_df.columns.tolist()}")
print("\n샘플 (처음 5행):")
print(final_df.head().to_string())
print("\n샘플 (마지막 5행):")
print(final_df.tail().to_string())
