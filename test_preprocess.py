from src.preprocess import load_and_aggregate, make_features
hourly, hourly_equip = load_and_aggregate('data/rtu_data_full.csv')
print(type(hourly))
hourly = make_features(hourly)
print('OK', hourly.shape)