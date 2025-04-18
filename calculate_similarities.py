import numpy as np

 from numba import jit, types, extending
 
 
 @jit(nopython=True)
 def calculate_distances(reference_features, features_query_i, metric):
     """
     :param reference_features: reference feature matrix for lag query i (number_of_reference,window_size)
     :param features_query_i: list of features in lag query i (len=window_size)
     :param metric: metric
     :return: array of distances (len_distances,1)
     """
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
 def get_k_similar_data(reference, features_query_i, k, metric):
     """
     get k similar features and related targets
     :param reference: reference matrix with features and targets for lag query i (number_of_reference,window_size+1)
     :param features_query_i: list of features in lag query i (len=window_size)
     :param k: number of similar examples
     :param metric: metric
     :return: k similar data
     """
     # Calculate distances between the query and all references
     reference_features = reference[:, :-1]
     distances_array = calculate_distances(reference_features, features_query_i, metric)
     reference_array_with_distance = add_new_column_to_array(reference, distances_array)
     # Sort DataFrame by distance
     ordered_similar_data = sort_array_by_distance(reference_array_with_distance)
     # select top k
     k_similar_data = ordered_similar_data[:k, :]
     return k_similar_data
 
 
 @extending.overload_method(types.Array, 'vstack')
 def get_k_similar_data_for_all_query(reference_features, reference_targets, query_features, query_targets, k, metric):
     """
     get k similar features and related targets for each window in query
     :param reference_features:
     :param reference_targets:
     :param query_features:
     :param query_targets:
     :param k:
     :param metric:
     :return: k similar data for all query
     """
     result_data = []
     reference = add_new_column_to_array(reference_features, reference_targets)
     query = add_new_column_to_array(query_features, query_targets)
     features_query_0 = query_features[0]
     k_similar_data = get_k_similar_data(reference, features_query_0, k, metric)
     result_data.append(k_similar_data)
     reference2 = add_rows_to_array(reference, query[1:])
 
     for i in range(1, len(query)):
         features_query_i = query_features[i]
         k_similar_data = get_k_similar_data(reference2[:len(reference) + i - 1], features_query_i, k, metric)
         result_data.append(k_similar_data)
 
     k_similar_data_for_all_query = np.asarray(result_data)
 
     return k_similar_data_for_all_query
