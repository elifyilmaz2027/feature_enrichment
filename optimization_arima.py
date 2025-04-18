from pmdarima.arima import auto_arima
import pandas as pd
import numpy as np

train = pd.read_csv('prepared_data_train_test/train_data.csv')
test = pd.read_csv('prepared_data_train_test/test_data.csv')
train_data = train[['TotalFlow']]
test_data = test[['TotalFlow']]
train_size = 11808
test_data = np.array(test_data["TotalFlow"])
train_data = np.array(train_data["TotalFlow"])[-train_size:]

model = auto_arima(train_data, start_p=0, start_q=0,
                   max_p=5, max_q=0,
                   d=0,
                   seasonal=False,
                   start_P=0,
                   D=None,
                   trace=True)


