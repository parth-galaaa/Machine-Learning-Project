from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score

regressors_results = pd.read_csv('./regressors_results.csv')
regressors_results['CNN'] = regressors_results['CNN'].str.strip('[]').astype(float)
regressors_results['LSTM'] = regressors_results['LSTM'].str.strip('[]').astype(float)
regressors_results['Hybrid'] = regressors_results['Hybrid'].str.strip('[]').astype(float)


y_true = regressors_results['TARGET DATA']


results = {}
for model in regressors_results.columns[1:]:
    y_pred = regressors_results[model]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    results[model] = {'RMSE': rmse, 'R2': r2}


results_df = pd.DataFrame(results).T
print(results_df)



