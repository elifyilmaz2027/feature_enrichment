import statsmodels.api as sm
import pandas as pd
from evaluation_metrics import *


def arimamodel(train_data, test_data):
    arima = sm.tsa.arima.ARIMA(train_data, order=(5, 0, 0))
    arima_fit = arima.fit()
    print(arima_fit.summary())
    parameters = arima_fit.params
    a1 = parameters[1]
    a2 = parameters[2]
    a3 = parameters[3]
    a4 = parameters[4]
    a5 = parameters[5]
    train_predictions = []
    for t in range(4, len(train_data)):
        output_train = (train_data[t - 4] * a5) + (train_data[t - 3] * a4) + (train_data[t - 2] * a3) + (
                    train_data[t - 1] * a2) + (train_data[t] * a1)
        train_predictions.append(output_train)

    test_data2 = []
    test_data2.append(train_data[-5])
    test_data2.append(train_data[-4])
    test_data2.append(train_data[-3])
    test_data2.append(train_data[-2])
    test_data2.append(train_data[-1])
    for i in range(len(test_data) - 1):
        test_data2.append(test_data[i])

    test_predictions = []
    for t in range(4, len(test_data2)):
        output_test = (test_data2[t - 4] * a5) + (test_data2[t - 3] * a4) + (test_data2[t - 2] * a3) + (
                    test_data2[t - 1] * a2) + (test_data2[t] * a1)
        test_predictions.append(output_test)

    return train_predictions, test_predictions


train = pd.read_csv('prepared_data_train_test/train_data.csv')
test = pd.read_csv('prepared_data_train_test/test_data.csv')
train_data = np.array(train[['TotalFlow']])[-11808:]
test_data = np.array(test[['TotalFlow']])

train_predictions, test_predictions = arimamodel(train_data, test_data)
df_train_preds = pd.DataFrame(train_predictions)
df_test_preds = pd.DataFrame(test_predictions)
df_train_preds.to_csv("predictions/train_arima.csv")
df_test_preds.to_csv("predictions/test_arima.csv")

mape_test = MAPE(test_data, test_predictions)
mape100_test = MAPE_100(test_data, test_predictions)
mape350_test = MAPE_350(test_data, test_predictions)
mae_test = MAE(test_data, test_predictions)

mape_train = MAPE(train_data[4:], train_predictions)
mape100_train = MAPE_100(train_data[4:], train_predictions)
mape350_train = MAPE_350(train_data[4:], train_predictions)
mae_train = MAE(train_data[4:], train_predictions)

print("Results Test")
print("MAPE:", mape_test)
print("MAPE_100:", mape100_test)
print("MAPE_350:", mape350_test)
print("MAE:", mae_test)

print("Results Train")
print("MAPE:", mape_train)
print("MAPE_100:", mape100_train)
print("MAPE_350:", mape350_train)
print("MAE:", mae_train)

