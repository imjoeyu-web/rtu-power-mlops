import pandas as pd
import numpy as np


def load_and_aggregate(filepath: str) -> pd.DataFrame:
    """
    CSV 로드 → 설비별 1시간 평균 → 전체 합산 → hourly_pow 생성
    """
    df = pd.read_csv(filepath)
    df['dt'] = pd.to_datetime(df['localtime'], format='%Y%m%d%H%M%S')

    hourly = (
        df.groupby(['module(equipment)', pd.Grouper(key='dt', freq='1H')])
        ['activePower'].mean()
        .reset_index()
        .groupby('dt')['activePower'].sum()
        .reset_index()
        .rename(columns={'activePower': 'hourly_pow'})
    )
    return hourly


def make_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    시간 피처 + lag + rolling 생성
    """
    df = hourly.copy()

    # 시간 피처
    df['hour']     = df['dt'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # lag 피처
    df['lag_1']   = df['hourly_pow'].shift(1)
    df['lag_24']  = df['hourly_pow'].shift(24)
    df['lag_168'] = df['hourly_pow'].shift(168)

    # rolling 피처
    df['rolling_mean_24']  = df['hourly_pow'].shift(1).rolling(24).mean()
    df['rolling_std_24']   = df['hourly_pow'].shift(1).rolling(24).std()
    df['rolling_mean_168'] = df['hourly_pow'].shift(1).rolling(168).mean()

    df = df.dropna().reset_index(drop=True)
    return df


FEATURES = [
    'hour', 'hour_sin', 'hour_cos',
    'lag_1', 'lag_24', 'lag_168',
    'rolling_mean_24', 'rolling_std_24', 'rolling_mean_168'
]
TARGET = 'hourly_pow'
