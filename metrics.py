import numpy as np
import math
from numba import jit


@jit(nopython=True)
def euclidean(v1, v2):
    difference_vector = np.subtract(v1, v2)
    sum_of_square_differences = 0
    for i in range(len(v1)):
        sum_of_square_differences += difference_vector[i] ** 2
    return math.sqrt(sum_of_square_differences)


@jit(nopython=True)
def weighted_euclidean(v1, v2):
    difference_vector = np.subtract(v1, v2)
    sum_of_square_differences = 0
    total_weights = (len(v1) * (len(v1) + 1)) / 2
    for i in range(len(v1)):
        sum_of_square_differences += ((i + 1) / total_weights) * (difference_vector[i] ** 2)
    return math.sqrt(sum_of_square_differences)
