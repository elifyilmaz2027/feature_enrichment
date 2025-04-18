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

for w in range(2, 21):
    window_size = w

    train_similar_trajectories = np.load(
        f"prepared_data_similar_trajectories/train_300_similar_data_for_all_query_with_window_size_{window_size}_and_metric_weighted_euclidean.npy")
    test_similar_trajectories = np.load(
        f"prepared_data_similar_trajectories/test_300_similar_data_for_all_query_with_window_size_{window_size}_and_metric_weighted_euclidean.npy")

    mapes_test = []
    mape100s_test = []
    mape350s_test = []
    maes_test = []

    mapes_train = []
    mape100s_train = []
    mape350s_train = []
    maes_train = []

    for k in range(1, 51):
        print(k)
        train_predictions, test_predictions = select_avg_k_similar_targets_train_test(train_similar_trajectories, test_similar_trajectories, k)

        mape_train = MAPE(real_values_train, train_predictions)
        mape100_train = MAPE_100(real_values_train, train_predictions)
        mape350_train = MAPE_350(real_values_train, train_predictions)
        mae_train = MAE(real_values_train, train_predictions)
        print("Mae train", mae_train)
        mapes_train.append(mape_train)
        mape100s_train.append(mape100_train)
        mape350s_train.append(mape350_train)
        maes_train.append(mae_train)

        mape_test = MAPE(real_values_test, test_predictions)
        mape100_test = MAPE_100(real_values_test, test_predictions)
        mape350_test = MAPE_350(real_values_test, test_predictions)
        mae_test = MAE(real_values_test, test_predictions)
        print("Mae test", mae_test)
        mapes_test.append(mape_test)
        mape100s_test.append(mape100_test)
        mape350s_test.append(mape350_test)
        maes_test.append(mae_test)

    results_train = pd.DataFrame()
    results_train["mae"] = maes_train
    results_train["mape"] = mapes_train
    results_train["mape100"] = mape100s_train
    results_train["mape350"] = mape350s_train

    results_test = pd.DataFrame()
    results_test["mae"] = maes_test
    results_test["mape"] = mapes_test
    results_test["mape100"] = mape100s_test
    results_test["mape350"] = mape350s_test

    results_train.to_csv(f"results_st/train_similar_trajectory_results_avg_k_similar_targets_window_size_{window_size}_weighted_euclidean.csv")
    results_test.to_csv(f"results_st/test_similar_trajectory_results_avg_k_similar_targets_window_size_{window_size}_weighted_euclidean.csv")

