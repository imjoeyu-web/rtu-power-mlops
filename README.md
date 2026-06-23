⚡ RTU Power MLOps

스마트팩토리 설비 전력 소비 예측 및 이상탐지 MLOps 파이프라인

프로젝트 개요

공장 RTU(Remote Terminal Unit) 센서 데이터를 기반으로 시간별 전력 소비량을 예측하고, 이상 구간을 탐지하는 엔드투엔드 MLOps 파이프라인입니다.

단국대학교 데이터 사이언스 경진대회 출품작 (Public Score: 0.3906)

주요 기능


전력 예측: LightGBM 기반 시간별 전력 소비량 재귀 예측 (5월 672시간)
이상탐지: Rolling Z-score + Isolation Forest 이중 탐지
실험 추적: MLflow로 파라미터/메트릭 자동 기록
대시보드: Streamlit 기반 실시간 모니터링
자동화: GitHub Actions CI/CD 파이프라인


프로젝트 구조

rtu-power-mlops/
├── src/
│   ├── preprocess.py   # 데이터 로드 & hourly 집계
│   ├── model.py        # LightGBM 학습 & 재귀 예측
│   ├── anomaly.py      # 이상탐지 (Z-score, Isolation Forest)
│   └── pipeline.py     # 전체 파이프라인 실행
├── dashboard/
│   └── app.py          # Streamlit 대시보드
├── notebooks/
│   ├── eda_rtu.ipynb   # 탐색적 데이터 분석
│   └── modeling_rtu.ipynb  # 모델링 실험
├── data/               # 데이터 (gitignore)
├── output/             # 결과물 (gitignore)
├── Dockerfile
├── requirements.txt
└── .github/workflows/pipeline.yml

데이터


출처: 단국대학교 데이터 사이언스 경진대회
기간: 2024.12 ~ 2025.04 (5개월)
크기: 33,696,013행 × 19컬럼
설비: 13개 모듈 (분쇄기, 호기, 분전반 등)
수집 주기: 5초


EDA 주요 인사이트


공장 24시간 풀가동 (operation 컬럼 전부 1)
요일/주말 패턴 없음 → dow, is_weekend 피처 제외
자기상관 없음 (ACF/PACF) → lag 피처 효과 제한적
설비별 소비 패턴 균일 → Bottom-Up 방식 효과 미미
currentR/S/T, reactivePowerLagging 상관 높으나 예측 시점에 사용 불가


모델링 전략

타겟: hourly_pow (시간별 공장 전체 전력 소비)
피처: hour_sin/cos, lag_1/24/168, rolling_mean/std_24/168
방식: 재귀 예측 (1시간씩 순서대로 예측 → lag에 반영)
모델: LightGBM

실행 방법

로컬 실행

bash# 파이프라인 실행
python -m src.pipeline

# MLflow 대시보드
mlflow ui
# http://localhost:5000

# Streamlit 대시보드
streamlit run dashboard/app.py
# http://localhost:8501

Docker 실행

bash# 이미지 빌드
docker build -t rtu-power-mlops .

# 컨테이너 실행
docker run -p 8501:8501 \
  -v ./data:/app/data \
  -v ./output:/app/output \
  rtu-power-mlops

# http://localhost:8501

결과

항목값Public Score0.3906Train SMAPE0.09%5월 누적 전력26,288,332 kWh5월 전기요금47.32 억원5월 탄소배출11,146,253 kgCO₂Z-score 이상탐지27건Isolation Forest 이상탐지35건

기술 스택

구분기술모델LightGBM, scikit-learn실험 추적MLflow대시보드Streamlit컨테이너DockerCI/CDGitHub Actions언어Python 3.11
