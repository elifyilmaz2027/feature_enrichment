import pandas as pd
import numpy as np


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




