import numpy as np


# Defining MAPE function
def MAPE(actual_values, predicted_values):
    predicted_values = np.array(predicted_values).reshape((len(predicted_values), 1))
    actual_values = np.array(actual_values).reshape((len(actual_values), 1))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
    return mape


# Defining MAPE_100 function
def MAPE_100(actual_values, predicted_values):
    predicted_values = np.array(predicted_values).reshape((len(predicted_values), 1))
    actual_values = np.array(actual_values).reshape((len(actual_values), 1))
    x = np.concatenate((actual_values, predicted_values), axis=1)
    x_100 = x[x[:, 0] > 100]
    mape = np.mean(np.abs((x_100[:, 0] - x_100[:, 1]) / x_100[:, 0])) * 100
    return mape


# Defining MAPE_350 function
def MAPE_350(actual_values, predicted_values):
    predicted_values = np.array(predicted_values).reshape((len(predicted_values), 1))
    actual_values = np.array(actual_values).reshape((len(actual_values), 1))
    x = np.concatenate((actual_values, predicted_values), axis=1)
    x_350 = x[x[:, 0] > 350]
    mape = np.mean(np.abs((x_350[:, 0] - x_350[:, 1]) / x_350[:, 0])) * 100
    return mape


# Defining MAE function
def MAE(actual_values, predicted_values):
    predicted_values = np.array(predicted_values).reshape((len(predicted_values), 1))
    actual_values = np.array(actual_values).reshape((len(actual_values), 1))
    mae = np.mean(np.abs(actual_values - predicted_values))
    return mae