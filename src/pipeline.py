"""
RTU 전력 예측 파이프라인
실행: python src/pipeline.py
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.preprocess import load_and_aggregate, make_features
from src.model import train_model, recursive_predict, make_submission
from src.anomaly import detect_zscore, detect_isolation_forest, get_anomaly_summary

def run_pipeline(data_path: str = 'gs://rtu-power-data/rtu_data_full.csv',
                 output_path: str = 'output/submission.csv'):

    print('=' * 50)
    print('RTU 전력 예측 파이프라인 시작')
    print('=' * 50)

    # 1. 전처리
    print('\n[1/4] 데이터 로드 & 전처리')
    hourly = load_and_aggregate(data_path)
    hourly = make_features(hourly)
    print(f'  학습 데이터: {hourly.shape[0]}시간')

    # 2. 모델 학습
    print('\n[2/4] 모델 학습')
    model = train_model(hourly)

    # 3. 5월 예측
    print('\n[3/4] 5월 재귀 예측')
    may_dates, preds = recursive_predict(model, hourly)

    # 4. 이상탐지
    print('\n[4/4] 이상탐지')
    hourly = detect_zscore(hourly)
    hourly = detect_isolation_forest(hourly)
    summary = get_anomaly_summary(hourly)

    # 5. 결과 저장
    print('\n[저장] 제출 파일 생성')
    submission = make_submission(may_dates, preds, output_path)

    # 이상탐지 결과 저장
    anomaly_path = output_path.replace('submission.csv', 'anomaly_result.csv')
    hourly[['dt', 'hourly_pow', 'z_score', 'zscore_anomaly', 'iso_anomaly']].to_csv(
        anomaly_path, index=False
    )
    print(f'이상탐지 결과 저장: {anomaly_path}')

    print('\n' + '=' * 50)
    print('파이프라인 완료!')
    print('=' * 50)

    return {
        'hourly': hourly,
        'model': model,
        'may_dates': may_dates,
        'preds': preds,
        'submission': submission
    }


if __name__ == '__main__':
    result = run_pipeline()
