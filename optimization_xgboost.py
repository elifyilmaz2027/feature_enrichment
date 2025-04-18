from xgboost import XGBRegressor
import pandas as pd
from evaluation_metrics import *
from sklearn.model_selection import RandomizedSearchCV

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
data4 = data3[-(train_size + test_size):].reset_index()
data4 = data4[['t-673', 't-672', 't-671', 't-97', 't-96', 't-95', 't-2', 't-1', 't']]


mapes = []
mape100s = []
mape350s = []
maes = []
real_values_test = np.array(test_data["TotalFlow"])
real_values_train = np.array(train_data["TotalFlow"])[-11808:]

values = data4.values
data = values[:, :]
train = data[:-test_size, :]
test = data[-test_size:, :]
train_features, train_targets = train[:, :-1], train[:, -1]
test_features, test_targets = test[:, :-1], test[:, -1]

parameters = {'objective': ['reg:squarederror'],
              'validate_parameters': [True, False],
              'booster': ['gbtree', 'gblinear'],
              'min_split_loss': [0, 1, 3, 5, 7, 9],
              'learning_rate': [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
              'n_estimators': [40, 50, 60, 70, 80, 90, 100, 150, 200],
              "reg_alpha": [0, 0.5, 1, 1.5, 2, 2.5, 3],
              "reg_lambda": [0, 0.5, 1, 1.5, 2, 2.5, 3]}

model = XGBRegressor()
random_search_xgb = RandomizedSearchCV(model, parameters, scoring='neg_mean_absolute_error')
random_search_xgb.fit(train_features, train_targets)

train_predictions = random_search_xgb.predict(train_features)
test_predictions = random_search_xgb.predict(test_features)

mape_test = MAPE(real_values_test, test_predictions)
mape100_test = MAPE_100(real_values_test, test_predictions)
mape350_test = MAPE_350(real_values_test, test_predictions)
mae_test = MAE(real_values_test, test_predictions)

mape_train = MAPE(real_values_train, train_predictions)
mape100_train = MAPE_100(real_values_train, train_predictions)
mape350_train = MAPE_350(real_values_train, train_predictions)
mae_train = MAE(real_values_train, train_predictions)

print(random_search_xgb.best_params_)
print("mape", mape_test)
print("mape100", mape100_test)
print("mape350", mape350_test)
print("mae", mae_test)
print("---Train---")
print("mape", mape_train)
print("mape100", mape100_train)
print("mape350", mape350_train)
print("mae", mae_train)

