import numpy as np
import matplotlib.pyplot as plt

data = np.load("prepared_data_similar_trajectories/train_300_similar_data_for_all_query_with_window_size_10_and_metric_weighted_euclidean.npy")
print(data.shape)

data_query_features = np.load("prepared_data_similar_trajectories/train_query_features_with_window_size_10.npy")

image_data = np.vstack([data_query_features[10], data[10][:50, :-2]])
plt.imshow(image_data)
plt.savefig('image.png', dpi=500)
