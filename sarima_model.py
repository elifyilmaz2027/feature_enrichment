import numpy as np
import statsmodels.api as sm
import pandas as pd
from evaluation_metrics import MAE, MAPE


def sarimamodel(train_data, test_data):
    data = np.concatenate([train_data, test_data])
    data2 = pd.DataFrame(data)
    data3 = pd.concat(
        [data2.shift(673), data2.shift(672), data2.shift(5), data2.shift(4), data2.shift(3), data2.shift(2), data2.shift(1), data2], axis=1)
    data3.columns = ['t-673', 't-672', 't-5', 't-4', 't-3', 't-2', 't-1', 't']
    data4 = data3.values
    train_size = len(train_data)
    train, test = data4[673:train_size], data4[train_size:]
    train_X, train_y = train[:, :7], train[:, -1]
    test_X, test_y = test[:, :7], test[:, -1]

    """from pmdarima.arima import auto_arima
    model = auto_arima(y=train_y, X=train_X, start_p=0, start_q=0,
                       max_p=5, max_q=0,
                       d=0,
                       seasonal=False,
                       start_P=0,
                       D=None,
                       trace=True)

    print("result")"""

    sarima = sm.tsa.arima.ARIMA(train_y, order=(0,0,0), exog=train_X)
    sarima_fit = sarima.fit()
    print(sarima_fit.summary())

    train_predictions = sarima_fit.predict(start=len(train_y), end=len(train_y) + len(train_y) - 1,
                                           dynamic=train_data.all(), exog=train_X)

    test_predictions = sarima_fit.predict(start=len(train_y), end=len(train_y) + len(test_y) - 1,
                                          dynamic=test_data.all(), exog=test_X)

    return train_predictions, test_predictions


train = pd.read_csv('prepared_data_train_test/train_data.csv')
test = pd.read_csv('prepared_data_train_test/test_data.csv')
train_data = np.array(train[['TotalFlow']])[-11808:]
test_data = np.array(test[['TotalFlow']])

train_predictions, test_predictions = sarimamodel(train_data, test_data)
df_train_preds = pd.DataFrame(train_predictions)
df_test_preds = pd.DataFrame(test_predictions)
df_train_preds.to_csv("predictions/train_sarima.csv")
df_test_preds.to_csv("predictions/test_sarima.csv")

mape_test = MAPE(test_data, test_predictions)
mae_test = MAE(test_data, test_predictions)

mape_train = MAPE(train_data[673:], train_predictions)
mae_train = MAE(train_data[673:], train_predictions)



print("Results Test")
print("MAPE:", mape_test)
print("MAE:", mae_test)

print("Results Train")
print("MAPE:", mape_train)
print("MAE:", mae_train)

