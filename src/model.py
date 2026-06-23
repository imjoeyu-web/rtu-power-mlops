import pandas as pd
import numpy as np
import lightgbm as lgb
from preprocess import FEATURES, TARGET


def train_model(df: pd.DataFrame) -> lgb.LGBMRegressor:
    """
    전체 데이터로 LightGBM 학습
    """
    X = df[FEATURES]
    y = df[TARGET]

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)
    print(f'학습 완료: {X.shape[0]}시간 데이터')
    return model


def recursive_predict(model: lgb.LGBMRegressor,
                      history: pd.DataFrame,
                      n_hours: int = 672) -> tuple:
    """
    재귀 예측: 1시간씩 순서대로 예측 → 예측값을 lag에 반영
    """
    may_dates = pd.date_range('2025-05-01', periods=n_hours, freq='1H')
    hist = history[['dt', 'hourly_pow']].copy()
    preds = []

    for dt in may_dates:
        hour     = dt.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        def get_val(h):
            row = hist[hist['dt'] == dt - pd.Timedelta(hours=h)]
            return row['hourly_pow'].values[0] if len(row) > 0 else np.nan

        lag_1   = get_val(1)
        lag_24  = get_val(24)
        lag_168 = get_val(168)

        recent_24  = hist.tail(24)['hourly_pow']
        recent_168 = hist.tail(168)['hourly_pow']

        X_pred = pd.DataFrame([[
            hour, hour_sin, hour_cos,
            lag_1, lag_24, lag_168,
            recent_24.mean(), recent_24.std(), recent_168.mean()
        ]], columns=FEATURES)

        pred = model.predict(X_pred)[0]
        preds.append(pred)

        new_row = pd.DataFrame({'dt': [dt], 'hourly_pow': [pred]})
        hist = pd.concat([hist, new_row], ignore_index=True)

    print(f'예측 완료: {len(preds)}시간 / 범위: {min(preds):.2f} ~ {max(preds):.2f}')
    return may_dates, preds


def make_submission(may_dates, preds: list, save_path: str = 'output/submission.csv'):
    """
    제출 파일 생성
    """
    agg_pow    = sum(preds)
    may_bill   = agg_pow * 180
    may_carbon = agg_pow * 0.424

    submission = pd.DataFrame({
        'id':         may_dates.strftime('%Y-%m-%d %H:%M:%S'),
        'hourly_pow': preds,
        'agg_pow':    agg_pow,
        'may_bill':   may_bill,
        'may_carbon': may_carbon
    })

    submission.to_csv(save_path, index=False)
    print(f'제출 파일 저장: {save_path}')
    print(f'  agg_pow:    {agg_pow:,.2f} kWh')
    print(f'  may_bill:   {may_bill:,.0f} 원')
    print(f'  may_carbon: {may_carbon:,.2f} kgCO₂')
    return submission
