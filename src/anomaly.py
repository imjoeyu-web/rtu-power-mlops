import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def detect_zscore(hourly: pd.DataFrame,
                  window: int = 24,
                  threshold: float = 2.5) -> pd.DataFrame:
    """
    Rolling Z-score 이상탐지
    직전 n시간 대비 통계적으로 벗어난 구간 탐지
    """
    df = hourly.copy()
    df['rolling_mean'] = df['hourly_pow'].rolling(window=window, min_periods=1).mean()
    df['rolling_std']  = df['hourly_pow'].rolling(window=window, min_periods=1).std()
    df['z_score']      = (df['hourly_pow'] - df['rolling_mean']) / df['rolling_std']
    df['zscore_anomaly'] = df['z_score'].abs() > threshold

    anomalies = df[df['zscore_anomaly']]
    print(f'[Z-score] 이상탐지: {len(anomalies)}건 / 전체 {len(df)}시간')
    return df


def detect_isolation_forest(hourly: pd.DataFrame,
                             contamination: float = 0.01) -> pd.DataFrame:
    """
    Isolation Forest 이상탐지
    다변량 패턴 기반 이상 탐지
    """
    df = hourly.copy()
    features = ['hourly_pow', 'lag_1', 'lag_24', 'rolling_mean_24']
    X = df[features].dropna()

    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    df.loc[X.index, 'iso_anomaly'] = iso.fit_predict(X)
    df['iso_anomaly'] = df['iso_anomaly'].map({1: False, -1: True})

    anomalies = df[df['iso_anomaly'] == True]
    print(f'[Isolation Forest] 이상탐지: {len(anomalies)}건 / 전체 {len(df)}시간')
    return df


def get_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    두 방법 모두 이상으로 탐지된 구간 (교집합)
    """
    if 'zscore_anomaly' not in df.columns or 'iso_anomaly' not in df.columns:
        print('이상탐지 먼저 실행하세요')
        return pd.DataFrame()

    both = df[df['zscore_anomaly'] & df['iso_anomaly']]
    print(f'[교집합] 두 방법 모두 이상: {len(both)}건')
    return both[['dt', 'hourly_pow', 'z_score']]
