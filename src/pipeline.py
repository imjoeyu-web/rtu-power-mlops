import sys
import os
import argparse
sys.path.append(os.path.dirname(__file__))

from src.preprocess import load_and_aggregate, make_features
from src.model import train_model, tune_and_train, recursive_predict, make_submission
from src.anomaly import (detect_zscore, detect_isolation_forest, get_anomaly_summary,
                         detect_zscore_per_equipment, detect_iso_per_equipment,
                         get_equip_anomaly_summary)


def run_pipeline(data_path='gs://rtu-power-data/rtu_sample_pipeline.csv',
                 output_path='output/submission.csv',
                 baseline=False,
                 n_trials=10):

    print('=' * 50)
    print('RTU pipeline start')
    print('=' * 50)

    print('\n[1/4] preprocess')
    hourly, hourly_equip = load_and_aggregate(data_path)
    hourly = make_features(hourly)
    print(f'  data: {hourly.shape[0]}h')
    print(f'  equipment: {hourly_equip["equipment"].nunique()}')

    print('\n[2/4] train')
    if baseline:
        model = train_model(hourly)
    else:
        model = tune_and_train(hourly, n_trials=n_trials)

    print('\n[3/4] predict')
    may_dates, preds = recursive_predict(model, hourly)

    print('\n[4/4] anomaly detection')
    hourly = detect_zscore(hourly)
    hourly = detect_isolation_forest(hourly)
    summary = get_anomaly_summary(hourly)

    equip_zscore = detect_zscore_per_equipment(hourly_equip)
    equip_iso = detect_iso_per_equipment(hourly_equip)
    equip_summary = get_equip_anomaly_summary(equip_zscore, equip_iso)

    print('\n[save]')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission = make_submission(may_dates, preds, output_path)

    anomaly_path = output_path.replace('submission.csv', 'anomaly_result.csv')
    hourly[['dt', 'hourly_pow', 'z_score', 'zscore_anomaly', 'iso_anomaly']].to_csv(anomaly_path, index=False)

    equip_anomaly_path = output_path.replace('submission.csv', 'anomaly_per_equipment.csv')
    equip_summary.to_csv(equip_anomaly_path, index=False)

    print('done!')
    return {'hourly': hourly, 'hourly_equip': hourly_equip, 'model': model,
            'may_dates': may_dates, 'preds': preds, 'submission': submission,
            'equip_summary': equip_summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--data', type=str, default='gs://rtu-power-data/rtu_sample_pipeline.csv')
    args = parser.parse_args()
    run_pipeline(data_path=args.data, baseline=args.baseline, n_trials=args.trials)