import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_data = pd.read_csv('prepared_data_train_test/test_data.csv')["TotalFlow"]

test_arima = pd.read_csv('predictions/test_arima.csv')["0"]
test_sarima = pd.read_csv('predictions/test_sarima.csv')["0"]
test_bg = pd.read_csv('predictions/test_bg_l_10_k_33.csv')["0"]
test_xgb = pd.read_csv('predictions/test_xgboost.csv')["0"]
test_bg_xgb1 = pd.read_csv('predictions/test_bg_xgboost_1_l_16_k_42.csv')["0"]
test_bg_xgb2 = pd.read_csv('predictions/test_bg_xgboost_2_l_14_k_34.csv')["0"]
test_naive = pd.read_csv('predictions/test_naive.csv')["0"]

plt.figure(figsize=(20,10))
plt.plot(test_data[:96], label='Gerçek Değer', marker="o")

plt.plot(test_xgb[:96], label='XGBoost', marker="o")

plt.plot(test_bg_xgb1[:96], label='BG-XGBoost I', marker="o")
plt.plot(test_bg_xgb2[:96], label='BG-XGBoost II', marker="o")

plt.ylabel("Trafik Akışı",fontsize=20)
plt.legend(fontsize=20)
plt.savefig('predictions_vs_actual_xgb_bgxgbs.png', dpi=500)
