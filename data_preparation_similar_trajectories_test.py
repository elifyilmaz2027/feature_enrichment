from calculate_similarities import *
from metrics import *
from preprocess import *
import datetime

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
    test_k_similar_data_for_all_query = get_k_similar_data_for_all_query(test_reference_features, test_reference_targets, test_query_features, test_query_targets, k, metric)
    print("Calculation Ended")
    print(datetime.datetime.now())
    file_name = f"prepared_data_similar_trajectories/test_{k}_similar_data_for_all_query_with_window_size_{window_size}_and_metric_{str_metric}.npy"
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
