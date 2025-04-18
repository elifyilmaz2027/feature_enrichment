import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates
from datetime import datetime
import matplotlib.dates as mdates

train_data = pd.read_csv('prepared_data_train_test/train_data.csv')
test_data = pd.read_csv('prepared_data_train_test/test_data.csv')


date_time = train_data["Timestamp"][-672:]
date_time = pd.to_datetime(date_time)
y_values = train_data["TotalFlow"][-672:]

DF = pd.DataFrame()
DF['value'] = y_values
DF = DF.set_index(date_time)
plt.figure(figsize=(10, 6))
plt.plot(DF, color="blue")
plt.xlabel("Zaman")
plt.ylabel("Trafik Akışı (araç sayısı / 15 dakika)")
#plt.gcf().autofmt_xdate()
date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
plt.gca().xaxis.set_major_formatter(date_format)
# Automatically format the x-axis labels
plt.gcf().autofmt_xdate()
plt.savefig('traffic_flow2.png', dpi=500)
