import pandas as pd
import numpy as np
from numba import jit, types, extending
import datetime
from metrics import *


def data_preprocess(file_path, window_size, interval_timestamps):
    """
    :param file_path: file path
    :param window_size: length of window
    :param interval_timestamps: general interval for data preparation
    :return: dataframe with features, timestamp and target
    """
    df = pd.read_csv(file_path)

    start_timestamp, end_timestamp = interval_timestamps[0], interval_timestamps[1]
    segment_data = df[(df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp)]
    segment_list = []
    for j in range(window_size, len(segment_data)):
        features = list(segment_data['TotalFlow'].iloc[j - window_size:j])
        target = segment_data['TotalFlow'].iloc[j]
        timestamp = segment_data['Timestamp'].iloc[j]
        segment_list.append({'Timestamp': timestamp, 'features': features, 'target': target})

    prepared_data = pd.DataFrame(segment_list)

    return prepared_data


def split_data_with_given_timestamps(prepared_data, timestamps):
    """
    :param prepared_data: output of data_process function
    :param timestamps: list of starting and ending timestamps for train and test references and queries b
    :return: tuple of train and test references and queries
    """
    train_reference_start, train_reference_end, train_query_start, train_query_end, \
        test_reference_start, test_reference_end, test_query_start, test_query_end = \
        timestamps[0], timestamps[1], timestamps[2], timestamps[3], timestamps[4], \
            timestamps[5], timestamps[6], timestamps[7]

    train_reference = prepared_data[
        (prepared_data['Timestamp'] >= train_reference_start) &
        (prepared_data['Timestamp'] <= train_reference_end)]
    train_query = prepared_data[
        (prepared_data['Timestamp'] >= train_query_start) &
        (prepared_data['Timestamp'] <= train_query_end)]

    test_reference = prepared_data[
        (prepared_data['Timestamp'] >= test_reference_start) &
        (prepared_data['Timestamp'] <= test_reference_end)]
    test_query = prepared_data[
        (prepared_data['Timestamp'] >= test_query_start) &
        (prepared_data['Timestamp'] <= test_query_end)]

    return train_reference, train_query, test_reference, test_query


def convert_df_to_array(train_reference, train_query, test_reference, test_query):
    """
    :param train_reference: train reference dataframe for train query, which includes features and target
    :param train_query: train query dataframe including features and target
    :param test_reference: test reference dataframe for test query, which includes features and targets
    :param test_query: test query dataframe including features and target
    :return: train-test features and targets for train-test references and queries
    """
    train_reference_features = train_reference["features"]
    train_reference_features = np.array([i for i in train_reference_features])
    train_reference_targets = np.array(train_reference["target"]).reshape((len(train_reference), 1))
    train_query_features = train_query["features"]
    train_query_features = np.array([i for i in train_query_features])
    train_query_targets = np.array(train_query["target"]).reshape((len(train_query), 1))

    test_reference_features = test_reference["features"]
    test_reference_features = np.array([i for i in test_reference_features])
    test_reference_targets = np.array(test_reference["target"]).reshape((len(test_reference), 1))
    test_query_features = test_query["features"]
    test_query_features = np.array([i for i in test_query_features])
    test_query_targets = np.array(test_query["target"]).reshape((len(test_query), 1))

    return train_reference_features, train_reference_targets, train_query_features, train_query_targets, test_reference_features, test_reference_targets, test_query_features, test_query_targets


@jit(nopython=True)
def calculate_distances_w_seasonality(reference_features, features_query_i, metric, seasonality: False):
    """
    :param reference_features: reference feature matrix for lag query i (number_of_reference,window_size)
    :param features_query_i: list of features in lag query i (len=window_size)
    :param metric: metric
    :param seasonality: if True, add daily seasonality; otherwise calculate default
    :return: array of distances (len_distances,1)
    """
    if seasonality:
        reference_features = reference_features[95::96, :]
    distances = []
    for i in range(len(reference_features)):
        distance = metric(reference_features[i], features_query_i)
        distances.append(distance)
    distances_array = np.array(distances)
    distances_array = distances_array.reshape((len(distances_array), 1))
    return distances_array


@extending.overload_method(types.Array, 'argsort')
def sort_array_by_distance(array):
    """
    sort array rows using last column
    :param array:
    :return: sorted array
    """
    return array[array[:, -1].argsort()]


@extending.overload_method(types.Array,'append')
def add_new_column_to_array(array1, array2):
    """
    concatenate arrays by axis 1
    :param array1:
    :param array2: array with shape (len(array),1)
    :return:
    """
    new_array = np.append(array1, array2, axis=1)
    return new_array


@extending.overload_method(types.Array, 'vstack')
def add_rows_to_array(array1, array2):
    """
    concatenate arrays by axis 0
    :param array1:
    :param array2:
    :return:
    """
    new_array = np.vstack([array1, array2])
    return new_array


@extending.overload_method(types.Array, 'vstack')
def get_k_similar_data(reference, features_query_i, k, metric, seasonality: False):
    """
    get k similar features and related targets
    :param reference: reference matrix with features and targets for lag query i (number_of_reference,window_size+1)
    :param features_query_i: list of features in lag query i (len=window_size)
    :param k: number of similar examples
    :param metric: metric
    :param seasonality: if True, add daily seasonality; otherwise calculate default
    :return: k similar data
    """
    # Calculate distances between the query and all references
    reference_features = reference[:, :-1]
    distances_array = calculate_distances_w_seasonality(reference_features, features_query_i, metric, seasonality=seasonality)
    if seasonality:
        reference = reference[95::96, :]
    reference_array_with_distance = add_new_column_to_array(reference, distances_array)
    # Sort DataFrame by distance
    ordered_similar_data = sort_array_by_distance(reference_array_with_distance)
    # select top k
    k_similar_data = ordered_similar_data[:k, :]
    return k_similar_data


@extending.overload_method(types.Array, 'vstack')
def get_k_similar_data_for_all_query(reference_features, reference_targets, query_features, query_targets, k, metric, seasonality: False):
    """
    get k similar features and related targets for each window in query
    :param reference_features:
    :param reference_targets:
    :param query_features:
    :param query_targets:
    :param k:
    :param metric:
    :param seasonality: if True, add daily seasonality; otherwise calculate default
    :return: k similar data for all query
    """
    result_data = []
    reference = add_new_column_to_array(reference_features, reference_targets)
    query = add_new_column_to_array(query_features, query_targets)
    features_query_0 = query_features[0]
    k_similar_data = get_k_similar_data(reference, features_query_0, k, metric, seasonality=seasonality)
    result_data.append(k_similar_data)
    reference2 = add_rows_to_array(reference, query[1:])

    for i in range(1, len(query)):
        features_query_i = query_features[i]
        k_similar_data = get_k_similar_data(reference2[:len(reference) + i - 1], features_query_i, k, metric, seasonality=seasonality)
        result_data.append(k_similar_data)

    k_similar_data_for_all_query = np.asarray(result_data)

    return k_similar_data_for_all_query


metric_map = {euclidean: "euclidean", weighted_euclidean: "weighted_euclidean"}


def main(file_path, window_size, interval_timestamps, timestamps, k, metric):
    """
    :param file_path:
    :param window_size:
    :param interval_timestamps:
    :param timestamps:
    :param k:
    :param metric:
    :return:
    """
    str_metric = metric_map[metric]
    prepared_data = data_preprocess(file_path, window_size, interval_timestamps)
    print("prepared data")
    train_reference, train_query, test_reference, test_query = split_data_with_given_timestamps(prepared_data,
                                                                                                timestamps)
    train_reference_features, train_reference_targets, train_query_features, train_query_targets, test_reference_features, test_reference_targets, test_query_features, test_query_targets = convert_df_to_array(
        train_reference, train_query, test_reference, test_query)
    print("obtained train and test ref and query matrices")

    print("Calculation Starting")
    print(datetime.datetime.now())
    train_k_similar_data_for_all_query = get_k_similar_data_for_all_query(train_reference_features, train_reference_targets, train_query_features, train_query_targets, k, metric, seasonality=True)
    print("Calculation Ended")
    print(datetime.datetime.now())
    file_name = f"prepared_data_similar_trajectories/train_{k}_similar_data_for_all_query_with_window_size_{window_size}_and_metric_{str_metric}_with_daily_seasonality.npy"
    np.save(file_name, train_k_similar_data_for_all_query)

    print("Calculation Starting for test")
    print(datetime.datetime.now())
    test_k_similar_data_for_all_query = get_k_similar_data_for_all_query(test_reference_features,
                                                                         test_reference_targets, test_query_features,
                                                                         test_query_targets, k, metric, seasonality=True)
    print("Calculation Ended for test")
    print(datetime.datetime.now())
    file_name = f"prepared_data_similar_trajectories/test_{k}_similar_data_for_all_query_with_window_size_{window_size}_and_metric_{str_metric}_with_daily_seasonality.npy"
    np.save(file_name, test_k_similar_data_for_all_query)


file_path = "data/d1_15minutes.csv"
interval_timestamps = ["2020-10-05 00:00:00", "2022-06-05 00:00:00"]
timestamps = ['2020-10-05 00:00:00', '2021-10-05 00:00:00',
              '2021-10-05 00:15:00', '2022-02-05 00:00:00',
              '2021-02-05 00:15:00', '2022-02-05 00:00:00',
              '2022-02-05 00:15:00', '2022-06-05 00:00:00']

k = 300
metric = weighted_euclidean
for w in range(2, 21):
    window_size = w
    main(file_path, window_size, interval_timestamps, timestamps, k, metric)


