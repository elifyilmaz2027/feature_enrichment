from numpy import asarray
from xgboost import XGBRegressor
import pandas as pd
import matplotlib.pyplot as plt
from evaluation_metrics import *


def train_test_split(data, n_test):
    """
    :param data: dataframe as a time series
    :param n_test: number of data in test set
    :return: tuple of train and test data
    """
    train = data[:-n_test, :]
    test = data[-n_test:, :]
    return train, test


def xgboost_forecast(train_data, test_features):
    train_data = asarray(train_data)
    train_features, train_targets = train_data[:, :-1], train_data[:, -1]
    model = XGBRegressor(validate_parameters=True, reg_lambda=0.5, reg_alpha=0.5, objective='reg:squarederror',
                         n_estimators=70, min_split_loss=9, learning_rate=0.1, booster='gbtree')
    model.fit(train_features, train_targets)
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    return train_predictions, test_predictions


def run_xgboost_model(data, n_test):
    train, test = train_test_split(data, n_test)
    train_data = [x for x in train]
    test_features, test_targets = test[:, :-1], test[:, -1]
    test_predictions = xgboost_forecast(train_data, test_features)
    return test_predictions


train_data = pd.read_csv('prepared_data_train_test/train_data.csv')
test_data = pd.read_csv('prepared_data_train_test/test_data.csv')
data = pd.concat([train_data, test_data])[['TotalFlow']]
data2 = pd.DataFrame(data.values)
data3 = pd.concat(
    [data2.shift(673), data2.shift(672), data2.shift(671), data2.shift(97), data2.shift(96), data2.shift(95),
     data2.shift(2), data2.shift(1), data2], axis=1)
data3.columns = ['t-673', 't-672', 't-671', 't-97', 't-96', 't-95', 't-2', 't-1', 't']
train_size = 11808
test_size = 11327
data4 = data3[-(train_size + test_size):].reset_index()
data4 = data4[['t-673', 't-672', 't-671', 't-97', 't-96', 't-95', 't-2', 't-1', 't']]


# add new features from similar trajectories
def select_avg_k_similar_targets_train_test(train_similar_trajectories, test_similar_trajectories, k):
    train_k_similar_targets = np.average(train_similar_trajectories[:, :k, -2], axis=1)
    test_k_similar_targets = np.average(test_similar_trajectories[:, :k, -2], axis=1)
    return train_k_similar_targets, test_k_similar_targets


def concat_train_test_avg_k_similar_targets(train_k_similar_targets, test_k_similar_targets):
    train_test_k_similar_targets = np.concatenate([train_k_similar_targets, test_k_similar_targets])
    return train_test_k_similar_targets


def add_avg_k_features_to_data(data, train_test_k_similar_targets):
    data['avg_target'] = train_test_k_similar_targets
    updated_data_with_k_features = data[[column for column in data.columns if column != 't'] + ['t']]
    return updated_data_with_k_features


real_values_test = np.array(test_data["TotalFlow"])
real_values_train = np.array(train_data["TotalFlow"])[-11808:]

train_similar_trajectories = np.load(
        f"prepared_data_similar_trajectories/train_300_similar_data_for_all_query_with_window_size_16_and_metric_weighted_euclidean.npy")
test_similar_trajectories = np.load(
        f"prepared_data_similar_trajectories/test_300_similar_data_for_all_query_with_window_size_16_and_metric_weighted_euclidean.npy")


train_k_similar_targets, test_k_similar_targets = select_avg_k_similar_targets_train_test(train_similar_trajectories, test_similar_trajectories, 42)
train_test_k_similar_targets = concat_train_test_avg_k_similar_targets(train_k_similar_targets, test_k_similar_targets)
updated_data_with_k_features = add_avg_k_features_to_data(data4, train_test_k_similar_targets)
values = updated_data_with_k_features.values
data = values[:, :]
train_predictions, test_predictions = run_xgboost_model(data, test_size)

df_train_preds = pd.DataFrame(train_predictions)
df_test_preds = pd.DataFrame(test_predictions)
df_train_preds.to_csv("predictions/train_bg_xgboost_1_l_16_k_42.csv")
df_test_preds.to_csv("predictions/test_bg_xgboost_1_l_16_k_42.csv")

mape_train = MAPE(real_values_train, train_predictions)
mape100_train = MAPE_100(real_values_train, train_predictions)
mape350_train = MAPE_350(real_values_train, train_predictions)
mae_train = MAE(real_values_train, train_predictions)
print("Mae train", mae_train)
print("Mape train", mape_train)


mape_test = MAPE(real_values_test, test_predictions)
mape100_test = MAPE_100(real_values_test, test_predictions)
mape350_test = MAPE_350(real_values_test, test_predictions)
mae_test = MAE(real_values_test, test_predictions)
print("Mae test", mae_test)
print("Mape test", mape_test)
