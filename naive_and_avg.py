import pandas as pd
from evaluation_metrics import *


def naive_method(data):
    data2 = pd.DataFrame(data.values)
    data3 = pd.concat([data2.shift(1), data2], axis=1)
    data3.columns = ['t-1', 't']
    data4 = data3.values
    train_size = 11808
    train, test = data4[-train_size:46849],  data4[46849:]
    train_predictions, train_values = train[:,0], train[:,1]
    test_predictions, test_values = test[:,0], test[:,1]
    return train_predictions, train_values, test_predictions, test_values


train_data = pd.read_csv('prepared_data_train_test/train_data.csv')
test_data = pd.read_csv('prepared_data_train_test/test_data.csv')
data = pd.concat([train_data, test_data])[['TotalFlow']]

train_predictions, train_actual, test_predictions, test_actual = naive_method(data)
print(test_predictions.shape)
df_train_preds = pd.DataFrame(train_predictions)
df_test_preds = pd.DataFrame(test_predictions)
df_train_preds.to_csv("predictions/train_naive.csv")
df_test_preds.to_csv("predictions/test_naive.csv")

mape_test = MAPE(test_actual, test_predictions)
mape100_test = MAPE_100(test_actual, test_predictions)
mape350_test = MAPE_350(test_actual, test_predictions)
mae_test = MAE(test_actual, test_predictions)

mape_train = MAPE(train_actual, train_predictions)
mape100_train = MAPE_100(train_actual, train_predictions)
mape350_train = MAPE_350(train_actual, train_predictions)
mae_train = MAE(train_actual, train_predictions)


print("Results Test Naive")
print("MAPE:", mape_test)
print("MAPE_100:", mape100_test)
print("MAPE_350:", mape350_test)
print("MAE:", mae_test)

print("Results Train Naive")
print("MAPE:", mape_train)
print("MAPE_100:", mape100_train)
print("MAPE_350:", mape350_train)
print("MAE:", mae_train)

