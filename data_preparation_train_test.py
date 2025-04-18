import pandas as pd
import numpy as np


def data_preprocess(file_path, interval_timestamps):
    """
    :param file_path: file path
    :param interval_timestamps: general interval for data preparation
    :return: dataframe with traffic flow and timestamp
    """
    df = pd.read_csv(file_path)

    start_timestamp, end_timestamp = interval_timestamps[0], interval_timestamps[1]
    prepared_data = df[(df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp)]
    prepared_data = prepared_data[["Timestamp", "TotalFlow"]]
    return prepared_data


def split_data_with_given_timestamps(prepared_data, timestamps):
    """
    :param prepared_data: output of data_process function
    :param timestamps: list of starting and ending timestamps for train and test
    :return: tuple of train and test dataframes
    """
    train_start, train_end, test_start, test_end = timestamps[0], timestamps[1], timestamps[2], timestamps[3]

    train_data = prepared_data[
        (prepared_data['Timestamp'] >= train_start) &
        (prepared_data['Timestamp'] <= train_end)]

    test_data = prepared_data[
        (prepared_data['Timestamp'] >= test_start) &
        (prepared_data['Timestamp'] <= test_end)]

    return train_data, test_data


file_path = "data/d1_15minutes.csv"
interval_timestamps = ["2020-10-05 00:00:00", "2022-06-05 00:00:00"]
timestamps = ['2020-10-05 00:00:00', '2022-02-05 00:00:00',
              '2022-02-05 00:15:00', '2022-06-05 00:00:00']

prepared_data = data_preprocess(file_path, interval_timestamps)
train_data, test_data = split_data_with_given_timestamps(prepared_data, timestamps)
train_data.to_csv("prepared_data_train_test/train_data.csv", index=False)
test_data.to_csv("prepared_data_train_test/test_data.csv", index=False)
