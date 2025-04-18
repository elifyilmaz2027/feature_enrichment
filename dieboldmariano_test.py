from dieboldmariano import dm_test
import pandas as pd
import numpy as np


test_data = np.array(pd.read_csv('prepared_data_train_test/test_data.csv')[['TotalFlow']])

naive = np.array(pd.read_csv('predictions/test_naive.csv')[['0']])
arima = np.array(pd.read_csv('predictions/test_arima.csv')[['0']])
xgboost = np.array(pd.read_csv('predictions/test_xgboost.csv')[['0']])
bg = np.array(pd.read_csv('predictions/test_bg_l_10_k_33.csv')[['0']])
bg_xgboost_1 = np.array(pd.read_csv('predictions/test_bg_xgboost_1_l_16_k_42.csv')[['0']])
bg_xgboost_2 = np.array(pd.read_csv('predictions/test_bg_xgboost_2_l_14_k_34.csv')[['0']])

print("Naive - BG Xgboost 1", dm_test(test_data, naive, bg_xgboost_1, one_sided=False))
print("Naive - BG Xgboost 2", dm_test(test_data, naive, bg_xgboost_2, one_sided=False))

print("Arima - BG Xgboost 1", dm_test(test_data, arima, bg_xgboost_1, one_sided=False))
print("Arima - BG Xgboost 2", dm_test(test_data, arima, bg_xgboost_2, one_sided=False))

print("XGBoost - BG Xgboost 1", dm_test(test_data, xgboost, bg_xgboost_1, one_sided=False))
print("XGBoost - BG Xgboost 2", dm_test(test_data, xgboost, bg_xgboost_2, one_sided=False))

print("BG - BG Xgboost 1", dm_test(test_data, bg, bg_xgboost_1, one_sided=False))
print("BG - BG Xgboost 2", dm_test(test_data, bg, bg_xgboost_2, one_sided=False))


print("BG Xgboost 1 - BG Xgboost 2", dm_test(test_data, bg_xgboost_1, bg_xgboost_2, one_sided=False))
