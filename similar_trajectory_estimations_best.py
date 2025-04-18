import numpy as np
import pandas as pd
from evaluation_metrics import *


train_data = pd.read_csv('prepared_data_train_test/train_data.csv')
test_data = pd.read_csv('prepared_data_train_test/test_data.csv')

train_size = 11808
test_size = 11327


def select_avg_k_similar_targets_train_test(train_similar_trajectories, test_similar_trajectories, k):
    train_k_similar_targets = np.average(train_similar_trajectories[:, :k, -2], axis=1)
    test_k_similar_targets = np.average(test_similar_trajectories[:, :k, -2], axis=1)
    return train_k_similar_targets, test_k_similar_targets


real_values_test = np.array(test_data["TotalFlow"])
real_values_train = np.array(train_data["TotalFlow"])[-train_size:]

train_similar_trajectories = np.load(
        f"prepared_data_similar_trajectories/train_300_similar_data_for_all_query_with_window_size_10_and_metric_weighted_euclidean.npy")
test_similar_trajectories = np.load(
        f"prepared_data_similar_trajectories/test_300_similar_data_for_all_query_with_window_size_10_and_metric_weighted_euclidean.npy")

train_predictions, test_predictions = select_avg_k_similar_targets_train_test(train_similar_trajectories, test_similar_trajectories, 33)

df_train_preds = pd.DataFrame(train_predictions)
df_test_preds = pd.DataFrame(test_predictions)
df_train_preds.to_csv("predictions/train_bg_l_10_k_33.csv")
df_test_preds.to_csv("predictions/test_bg_l_10_k_33.csv")

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

