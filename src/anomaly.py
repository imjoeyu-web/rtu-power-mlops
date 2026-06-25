import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# ── 1. 전체 합산 Z-score (기존 유지) ─────────────────────────────────
def detect_zscore(hourly: pd.DataFrame,
                  window: int = 24,
                  threshold: float = 2.5) -> pd.DataFrame:
    """
    Rolling Z-score 이상탐지 (전체 합산)
    """
    df = hourly.copy()
    df['rolling_mean'] = df['hourly_pow'].rolling(window=window, min_periods=1).mean()
    df['rolling_std']  = df['hourly_pow'].rolling(window=window, min_periods=1).std()
    df['z_score']      = (df['hourly_pow'] - df['rolling_mean']) / df['rolling_std']
    df['zscore_anomaly'] = df['z_score'].abs() > threshold

    anomalies = df[df['zscore_anomaly']]
    print(f'[Z-score] 이상탐지: {len(anomalies)}건 / 전체 {len(df)}시간')
    return df


# ── 2. 전체 합산 Isolation Forest (기존 유지) ────────────────────────
def detect_isolation_forest(hourly: pd.DataFrame,
                             contamination: float = 0.01) -> pd.DataFrame:
    """
    Isolation Forest 이상탐지 (전체 합산)
    """
    df = hourly.copy()
    features = ['hourly_pow', 'lag_1', 'lag_24', 'rolling_mean_24']
    X = df[features].dropna()

    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    df.loc[X.index, 'iso_anomaly'] = iso.fit_predict(X)
    df['iso_anomaly'] = df['iso_anomaly'].map({1: False, -1: True})

    anomalies = df[df['iso_anomaly'] == True]
    print(f'[Isolation Forest] 이상탐지: {len(anomalies)}건 / 전체 {len(df)}시간')
    return df


# ── 3. 설비별 Z-score ────────────────────────────────────────────────
def detect_zscore_per_equipment(hourly_equip: pd.DataFrame,
                                window: int = 24,
                                threshold: float = 2.5) -> pd.DataFrame:
    """
    설비별 Rolling Z-score 이상탐지
    각 설비의 개별 패턴 기준으로 탐지 → "3번 설비 이상" 수준으로 세분화
    """
    results = []

    for equip, group in hourly_equip.groupby('equipment'):
        g = group.copy().sort_values('dt').reset_index(drop=True)
        g['rolling_mean'] = g['hourly_pow'].rolling(window=window, min_periods=1).mean()
        g['rolling_std']  = g['hourly_pow'].rolling(window=window, min_periods=1).std()
        g['z_score']      = (g['hourly_pow'] - g['rolling_mean']) / g['rolling_std']
        g['zscore_anomaly'] = g['z_score'].abs() > threshold
        results.append(g)

    result_df = pd.concat(results, ignore_index=True)

    total     = len(result_df)
    anomalies = result_df['zscore_anomaly'].sum()
    print(f'[설비별 Z-score] 이상탐지: {int(anomalies)}건 / 전체 {total}시간')

    # 설비별 이상 건수 요약
    summary = (
        result_df[result_df['zscore_anomaly']]
        .groupby('equipment')
        .size()
        .reset_index(name='anomaly_count')
        .sort_values('anomaly_count', ascending=False)
    )
    if not summary.empty:
        print('[설비별 이상 건수]')
        print(summary.to_string(index=False))

    return result_df


# ── 4. 설비별 Isolation Forest ───────────────────────────────────────
def detect_iso_per_equipment(hourly_equip: pd.DataFrame,
                              contamination: float = 0.01) -> pd.DataFrame:
    """
    설비별 Isolation Forest 이상탐지
    """
    results = []

    for equip, group in hourly_equip.groupby('equipment'):
        g = group.copy().sort_values('dt').reset_index(drop=True)

        # lag/rolling 피처 생성
        g['lag_1']        = g['hourly_pow'].shift(1)
        g['lag_24']       = g['hourly_pow'].shift(24)
        g['rolling_mean'] = g['hourly_pow'].shift(1).rolling(24).mean()

        features = ['hourly_pow', 'lag_1', 'lag_24', 'rolling_mean']
        X = g[features].dropna()

        if len(X) < 10:
            g['iso_anomaly'] = False
        else:
            iso = IsolationForest(contamination=contamination,
                                  random_state=42, n_estimators=100)
            preds = iso.fit_predict(X)
            g.loc[X.index, 'iso_anomaly'] = preds == -1
            g['iso_anomaly'] = g['iso_anomaly'].fillna(False)

        results.append(g)

    result_df = pd.concat(results, ignore_index=True)

    anomalies = result_df['iso_anomaly'].sum()
    print(f'[설비별 Isolation Forest] 이상탐지: {int(anomalies)}건 / 전체 {len(result_df)}시간')

    summary = (
        result_df[result_df['iso_anomaly']]
        .groupby('equipment')
        .size()
        .reset_index(name='anomaly_count')
        .sort_values('anomaly_count', ascending=False)
    )
    if not summary.empty:
        print('[설비별 이상 건수]')
        print(summary.to_string(index=False))

    return result_df


# ── 5. 교집합 요약 ────────────────────────────────────────────────────
def get_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    두 방법 모두 이상으로 탐지된 구간 (교집합) - 전체 합산용
    """
    if 'zscore_anomaly' not in df.columns or 'iso_anomaly' not in df.columns:
        print('이상탐지 먼저 실행하세요')
        return pd.DataFrame()

    both = df[df['zscore_anomaly'] & df['iso_anomaly']]
    print(f'[교집합] 두 방법 모두 이상: {len(both)}건')
    return both[['dt', 'hourly_pow', 'z_score']]


def get_equip_anomaly_summary(zscore_df: pd.DataFrame,
                               iso_df: pd.DataFrame) -> pd.DataFrame:
    """
    설비별 두 방법 교집합 요약
    """
    merged = zscore_df[['equipment', 'dt', 'hourly_pow', 'z_score', 'zscore_anomaly']].merge(
        iso_df[['equipment', 'dt', 'iso_anomaly']],
        on=['equipment', 'dt'], how='inner'
    )
    both = merged[merged['zscore_anomaly'] & merged['iso_anomaly']]
    print(f'[설비별 교집합] 두 방법 모두 이상: {len(both)}건')

    summary = (
        both.groupby('equipment')
        .size()
        .reset_index(name='anomaly_count')
        .sort_values('anomaly_count', ascending=False)
    )
    if not summary.empty:
        print(summary.to_string(index=False))

    return both