import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from preprocess import FEATURES, TARGET
except ImportError:
    from src.preprocess import FEATURES, TARGET


# ── 1. Optuna 목적함수 ────────────────────────────────────────────────
def _objective(trial, df: pd.DataFrame) -> float:
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbose': -1,
    }

    tscv = TimeSeriesSplit(n_splits=3)
    X, y = df[FEATURES], df[TARGET]
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)])
        pred = model.predict(X_val)
        scores.append(mean_absolute_error(y_val, pred))

    return np.mean(scores)


# ── 2. 튜닝 + 최종 학습 ───────────────────────────────────────────────
def tune_and_train(df: pd.DataFrame,
                   n_trials: int = 10,
                   experiment_name: str = 'rtu-power-mlops') -> lgb.LGBMRegressor:
    """
    Optuna로 하이퍼파라미터 탐색 → 각 트라이얼 MLflow 기록
    → 최적 파라미터로 전체 데이터 최종 학습
    """
    mlflow.set_experiment(experiment_name)

    # -- Optuna 탐색
    print(f'[Optuna] {n_trials}회 탐색 시작...')
    study = optuna.create_study(direction='minimize')

    with mlflow.start_run(run_name='optuna_study', nested=False) as parent_run:
        mlflow.log_param('n_trials', n_trials)
        mlflow.log_param('cv_folds', 3)
        mlflow.log_param('features', FEATURES)

        def mlflow_callback(study, trial):
            with mlflow.start_run(run_name=f'trial_{trial.number}',
                                  nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric('val_mae', trial.value)

        study.optimize(
            lambda trial: _objective(trial, df),
            n_trials=n_trials,
            callbacks=[mlflow_callback],
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_mae    = study.best_value
        print(f'[Optuna] 최적 MAE: {best_mae:.4f}')
        print(f'[Optuna] 최적 파라미터: {best_params}')

        # -- 최종 학습 (전체 데이터)
        best_params.update({'random_state': 42, 'verbose': -1})
        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(df[FEATURES], df[TARGET])

        mlflow.log_params({f'best_{k}': v for k, v in best_params.items()})
        mlflow.log_metric('best_val_mae', best_mae)
        mlflow.lightgbm.log_model(final_model, artifact_path='model')

        print(f'[MLflow] run_id: {parent_run.info.run_id}')

    print(f'학습 완료: {df.shape[0]}시간 데이터')
    return final_model


# ── 3. 베이스라인 학습 (기존 호환용) ─────────────────────────────────
def train_model(df: pd.DataFrame) -> lgb.LGBMRegressor:
    """
    MLflow 기록 포함 베이스라인 학습 (파라미터 튜닝 없음)
    """
    mlflow.set_experiment('rtu-power-mlops')
    with mlflow.start_run(run_name='baseline'):
        params = dict(n_estimators=500, learning_rate=0.05,
                      num_leaves=31, random_state=42, verbose=-1)
        mlflow.log_params(params)

        # CV MAE 계산
        tscv = TimeSeriesSplit(n_splits=3)
        X, y = df[FEATURES], df[TARGET]
        scores = []
        for train_idx, val_idx in tscv.split(X):
            m = lgb.LGBMRegressor(**params)
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = m.predict(X.iloc[val_idx])
            scores.append(mean_absolute_error(y.iloc[val_idx], pred))
        val_mae = np.mean(scores)
        mlflow.log_metric('val_mae', val_mae)
        print(f'[Baseline] CV MAE: {val_mae:.4f}')

        # 전체 데이터로 최종 학습
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        mlflow.lightgbm.log_model(model, artifact_path='model')

    print(f'학습 완료: {df.shape[0]}시간 데이터')
    return model


# ── 4. 재귀 예측 ─────────────────────────────────────────────────────
def recursive_predict(model: lgb.LGBMRegressor,
                      history: pd.DataFrame,
                      n_hours: int = 672) -> tuple:
    """
    재귀 예측: 1시간씩 순서대로 예측 → 예측값을 lag에 반영
    """
    may_dates = pd.date_range('2025-05-01', periods=n_hours, freq='1h')
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


# ── 5. 제출 파일 생성 ─────────────────────────────────────────────────
def make_submission(may_dates, preds: list,
                    save_path: str = 'output/submission.csv'):
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
