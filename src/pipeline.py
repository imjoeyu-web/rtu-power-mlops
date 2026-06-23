"""
RTU 전력 예측 파이프라인
실행: python -m src.pipeline
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(__file__))

import mlflow
import mlflow.lightgbm
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.preprocess import load_and_aggregate, make_features, FEATURES, TARGET
from src.model import train_model, recursive_predict, make_submission
from src.anomaly import detect_zscore, detect_isolation_forest, get_anomaly_summary


def run_pipeline(data_path: str = 'data/rtu_data_full.csv',
                 output_path: str = 'output/submission.csv'):

    print('=' * 50)
    print('RTU 전력 예측 파이프라인 시작')
    print('=' * 50)

    # MLflow 실험 설정
    mlflow.set_experiment('RTU_전력예측')

    with mlflow.start_run(run_name='LightGBM_재귀예측'):

        # 1. 전처리
        print('\n[1/4] 데이터 로드 & 전처리')
        hourly = load_and_aggregate(data_path)
        hourly = make_features(hourly)
        print(f'  학습 데이터: {hourly.shape[0]}시간')

        # MLflow 데이터 정보 기록
        mlflow.log_param('data_path', data_path)
        mlflow.log_param('train_hours', hourly.shape[0])
        mlflow.log_param('features', FEATURES)

        # 2. 모델 학습
        print('\n[2/4] 모델 학습')
        model = train_model(hourly)

        # MLflow 모델 파라미터 기록
        mlflow.log_param('n_estimators', 500)
        mlflow.log_param('learning_rate', 0.05)
        mlflow.log_param('num_leaves', 31)

        # 학습 데이터 성능 (자기 자신 예측)
        y_train = hourly[TARGET]
        y_pred_train = model.predict(hourly[FEATURES])
        train_mae  = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_smape = np.mean(
            2 * np.abs(y_pred_train - y_train) /
            (np.abs(y_pred_train) + np.abs(y_train))
        ) * 100

        mlflow.log_metric('train_mae', train_mae)
        mlflow.log_metric('train_rmse', train_rmse)
        mlflow.log_metric('train_smape', train_smape)
        print(f'  Train MAE: {train_mae:.2f} / SMAPE: {train_smape:.4f}%')

        # 3. 5월 예측
        print('\n[3/4] 5월 재귀 예측')
        may_dates, preds = recursive_predict(model, hourly)

        agg_pow    = sum(preds)
        may_bill   = agg_pow * 180
        may_carbon = agg_pow * 0.424

        mlflow.log_metric('agg_pow', agg_pow)
        mlflow.log_metric('may_bill', may_bill)
        mlflow.log_metric('may_carbon', may_carbon)
        mlflow.log_metric('pred_min', min(preds))
        mlflow.log_metric('pred_max', max(preds))

        # 4. 이상탐지
        print('\n[4/4] 이상탐지')
        hourly = detect_zscore(hourly)
        hourly = detect_isolation_forest(hourly)
        summary = get_anomaly_summary(hourly)

        zscore_count = hourly['zscore_anomaly'].sum()
        iso_count    = hourly['iso_anomaly'].sum()
        both_count   = len(summary)

        mlflow.log_metric('anomaly_zscore_count', int(zscore_count))
        mlflow.log_metric('anomaly_iso_count', int(iso_count))
        mlflow.log_metric('anomaly_both_count', int(both_count))

        # 5. 결과 저장
        print('\n[저장] 제출 파일 생성')
        submission = make_submission(may_dates, preds, output_path)

        anomaly_path = output_path.replace('submission.csv', 'anomaly_result.csv')
        hourly[['dt', 'hourly_pow', 'z_score', 'zscore_anomaly', 'iso_anomaly']].to_csv(
            anomaly_path, index=False
        )
        print(f'이상탐지 결과 저장: {anomaly_path}')

        # MLflow 아티팩트 저장
        mlflow.log_artifact(output_path)
        mlflow.log_artifact(anomaly_path)
        mlflow.lightgbm.log_model(model, 'model')

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
