from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
# Remove brackets from the CNN, LSTM, and Hybrid columns and convert to float for calculations
regressors_results = pd.read_csv('./regressors_results.csv')
regressors_results['CNN'] = regressors_results['CNN'].str.strip('[]').astype(float)
regressors_results['LSTM'] = regressors_results['LSTM'].str.strip('[]').astype(float)
regressors_results['Hybrid'] = regressors_results['Hybrid'].str.strip('[]').astype(float)

# Extract target data
y_true = regressors_results['TARGET DATA']

# Calculate RMSE and R2 for each regression model
results = {}
for model in regressors_results.columns[1:]:
    y_pred = regressors_results[model]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    results[model] = {'RMSE': rmse, 'R2': r2}

# Convert results to a DataFrame for better readability
results_df = pd.DataFrame(results).T
print(results_df)


df = pd.read_csv('regressors_results.csv', sep=',')

for col in df.columns[1:]:  # Assuming first column is 'TARGET DATA'
    df[col] = df[col].apply(lambda x: float(x.strip('[]')) if isinstance(x, str) else x)

# Define a color map for different models
colors = ['skyblue', 'orange', 'lightgreen', 'red', 'violet', 'brown', 'pink']

# Create individual plots for each model
for i, model in enumerate(df.columns[1:]):  # Excluding 'TARGET DATA'
    plt.figure(figsize=(8, 6))
    plt.scatter(df['TARGET DATA'], df[model], color=colors[i % len(colors)], alpha=0.6, label=model)
    plt.plot(df['TARGET DATA'], df['TARGET DATA'], 'k--', label='Perfect Prediction')  # 45-degree line
    plt.xlabel('Target Data')
    plt.ylabel('Predicted Values')
    plt.title(f'Target Data vs. Predictions for {model}')
    plt.legend()
    plt.show()

# Calculate and display RMSE and R² scores for each model
rmse = {}
r2_scores = {}

for model in df.columns[1:]:  # Excluding 'TARGET DATA'
    rmse[model] = root_mean_squared_error(df['TARGET DATA'], df[model])
    r2_scores[model] = r2_score(df['TARGET DATA'], df[model])

# RMSE Bar Chart for each model
plt.figure(figsize=(10, 6))
plt.bar(rmse.keys(), rmse.values(), color='skyblue')
plt.title('RMSE of Different Models')
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.show()

# R² Score Bar Chart for each model
plt.figure(figsize=(10, 6))
plt.bar(r2_scores.keys(), r2_scores.values(), color='lightgreen')
plt.title('R² Scores of Different Models')
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.show()