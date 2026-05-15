# Konetic 프로젝트 — Claude Code 공유 컨텍스트

## 프로젝트 개요
**한국남동발전 AX 아이디어 경진대회** — 데이터 분석 부문  
화력발전소 대기오염물질(SOx, NOx, 먼지) 배출 예측 및 연료 최적화

---

## 데이터 현황

### 원천 데이터 (`data/raw/`)
| 파일 | 내용 | 주기 |
|------|------|------|
| `한국남동발전_대기오염물질배출농도(일평균).xls` | SOx/NOx/먼지 농도+유량 | 일 |
| `한국남동발전_기상정보(일평균).xls` | 온도·습도·풍향·풍속 등 | 일 |
| `한국남동발전_연료소비실적.xls` | 유연탄·LNG·유류 소비량 | 월 |
| `한국남동발전_발전실적.xls` | 발전량·열효율·이용률 | 월 |
| `every/태양광·풍력·해양소수력·연료전지_YYYYMM.xlsx` | 시간별 신재생 발전량 | 시간 |
| `thermal_gen_20220101_20260430_full.csv` | 삼천포 API 일별 발전량 | 일 |
| `thermal_gen_20220101_20260430_yeonghung.csv` | 영흥 API 일별 발전량 | 일 |
| `weather_삼천포_2020-07-01_2026-04-30.csv` | Open-Meteo 삼천포 기상 | 일 |
| `weather_영흥_2020-07-01_2026-04-30.csv` | Open-Meteo 영흥 기상 | 일 |

### 가공 데이터 (`data/processed/`)
| 파일 | 사업소 | 기간 | 행수 | 비고 |
|------|--------|------|-----:|------|
| `master_삼천포.csv` | 삼천포 화력 | 2020-07-16~2026-05-09 | 2,123 | 석탄, is_coal=1 |
| `master_영흥.csv` | 영흥 화력 | 2020-07-16~2026-05-09 | 2,121 | 석탄, is_coal=1 |
| `master_분당.csv` | 분당 복합 | 2022-01-01~2026-04-30 | 1,497 | LNG, is_coal=0, +fuel_cell_mwh |
| `master_dataset.csv` | 삼천포 (동일) | — | 2,123 | 하위 호환용, 수정 금지 |

---

## 사업소별 특성 (모델링 시 필수 참고)

| 항목 | 삼천포 | 영흥 | 분당 |
|------|--------|------|------|
| 주연료 | 유연탄(석탄) | 유연탄(석탄) | LNG |
| 주오염물질 | SOx·NOx·먼지 | SOx·NOx·먼지 | NOx 위주 (SOx≈0) |
| 신재생 | 태양광·풍력·해양소수력 | 태양광·풍력·해양소수력 | 연료전지 |
| 발전규모 | ~28,000 MWh/일 | ~67,000 MWh/일 | — |
| coal_ratio | ≈ 1.0 | ≈ 1.0 | ≈ 0.0 |

---

## 코드 구조

```
src/
├── data_loader.py   # API 수집, XLS 로딩, Open-Meteo, PLANT_COORDS/PLANT_CODES
├── features.py      # TARGET_COLS, 피처 그룹 상수, build_features(), get_X_y()
├── model.py         # 모델 학습/평가
└── optimizer.py     # 연료 최적화

notebooks/
├── 00_preprocessing.ipynb  # master_*.csv 생성 (PLANT 변수로 사업소 선택)
├── 01_eda.ipynb
├── 02_modeling.ipynb
└── 03_optimization.ipynb

data/
├── raw/             # 원천 데이터 (수정 금지)
└── processed/       # 가공 데이터

reports/             # 차트, 모델 pkl, 결과 CSV
```

---

## 핵심 상수 (features.py)

```python
TARGET_COLS = ["SOx", "NOx", "먼지"]   # 예측 타겟 (kg/일)

WEATHER_FEATURES = ["temp_mean", "humidity_mean", "wind_sin", "wind_cos",
                    "wind_speed_mean", "wind_speed_max", "precipitation",
                    "pressure_mean", "solar_radiation", "stagnation_idx"]

FUEL_FEATURES    = ["유연탄"]           # 석탄 발전소만 유효
GENERATION_FEATURES = ["gen_mwh_combined", "utilization", "heat_efficiency"]
```

---

## 배출량 산정 공식

```
일 배출량(kg) = 농도(mg/Sm³) × 유량(Sm³/min) × 1,440 min/일 ÷ 1,000,000
→ master_*.csv의 SOx/NOx/먼지 단위: kg/일
```

※ 원천 데이터 유량 단위: Sm³/min (× 86,400은 오류 — 2025년 5월 수정 완료)

---

## 작업 규칙

- `data/raw/` 파일은 **읽기 전용** — 절대 수정 금지
- `master_dataset.csv`는 **수정 금지** (하위 호환)
- 삼천포·영흥 분석은 `master_삼천포.csv`, `master_영흥.csv` 사용
- 분당은 LNG 복합이므로 `coal_ratio`, `유연탄` 피처 모델에서 제외
- 보고서용 차트는 `reports/` 에 저장, DPI=150 이상
