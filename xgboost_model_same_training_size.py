from numpy import asarray
from xgboost import XGBRegressor
import pandas as pd
from evaluation_metrics import *


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
    train = asarray(train)
    trainX, trainy = train[:, :-1], train[:, -1]
    model = XGBRegressor(validate_parameters=False, reg_lambda=1.5, reg_alpha=1, objective='reg:squarederror',
                         n_estimators=100, min_split_loss=7, learning_rate=0.05, booster='gbtree')
    model.fit(trainX, trainy)
    yhat = model.predict(testX)
    return yhat


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    train, test = train_test_split(data, n_test)
    history = [x for x in train]
    train = asarray(train)
    trainX, trainy = train[:, :-1], train[:, -1]
    testX, testy = test[:, :-1], test[:, -1]
    train_predictions = xgboost_forecast(history, trainX)
    test_predictions = xgboost_forecast(history, testX)
    return train_predictions, test_predictions


train_data = pd.read_csv('prepared_data_train_test/train_data.csv')
test_data = pd.read_csv('prepared_data_train_test/test_data.csv')
data = pd.concat([train_data, test_data])[['TotalFlow']]
data2 = pd.DataFrame(data.values)
data3 = pd.concat(
    [data2.shift(673), data2.shift(672), data2.shift(671), data2.shift(97), data2.shift(96), data2.shift(95),
     data2.shift(2), data2.shift(1), data2], axis=1)
data3.columns = ['t-673', 't-672', 't-671', 't-97', 't-96', 't-95', 't-2', 't-1', 't']
train_size = 11808
test_size = 11327

values = data3.values
data = values[-(train_size+test_size):, :]
test_actual = np.array(test_data["TotalFlow"])
train_actual = np.array(train_data["TotalFlow"])[-train_size:]

train_predictions, test_predictions = walk_forward_validation(data, test_size)

df_train_preds = pd.DataFrame(train_predictions)
df_test_preds = pd.DataFrame(test_predictions)
df_train_preds.to_csv("predictions/train_xgboost.csv")
df_test_preds.to_csv("predictions/test_xgboost.csv")

mape_test = MAPE(test_actual, test_predictions)
mape100_test = MAPE_100(test_actual, test_predictions)
mape350_test = MAPE_350(test_actual, test_predictions)
mae_test = MAE(test_actual, test_predictions)

mape_train = MAPE(train_actual, train_predictions)
mape100_train = MAPE_100(train_actual, train_predictions)
mape350_train = MAPE_350(train_actual, train_predictions)
mae_train = MAE(train_actual, train_predictions)

print("XGBoost Results Test")
print("MAPE:", mape_test)
print("MAPE_100:", mape100_test)
print("MAPE_350:", mape350_test)
print("MAE:", mae_test)

print("XGBoost Results Train")
print("MAPE:", mape_train)
print("MAPE_100:", mape100_train)
print("MAPE_350:", mape350_train)
print("MAE:", mae_train)


