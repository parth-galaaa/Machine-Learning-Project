import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score

# Read data from the CSV file
df = pd.read_csv('regressors_results.csv', sep=',')

# Convert columns with list values to floats
for col in df.columns[1:]:  # Assuming first column is 'TARGET DATA'
    df[col] = df[col].apply(lambda x: float(x.strip('[]')) if isinstance(x, str) else x)

# Apply a rolling average for smoothing
window_size = 100  # Adjust the window size as needed for smoothing
df_smooth = df.rolling(window=window_size).mean()

# Downsample the data (for readability)
downsample_rate = 10
df_smooth = df_smooth[::downsample_rate]

# Define a color map for different models
colors = ['skyblue', 'orange', 'lightgreen', 'red', 'violet', 'brown', 'pink']

# Plot all models and target data in a single figure with smoothing
plt.figure(figsize=(12, 6))
plt.plot(df_smooth.index, df_smooth['TARGET DATA'], color='black', linestyle='--', label='Target Data')  # Target reference line

for i, model in enumerate(df.columns[1:]):  # Excluding 'TARGET DATA'
    plt.plot(df_smooth.index, df_smooth[model], color=colors[i % len(colors)], label=model)  # Model prediction line

plt.xlabel('Data Index')
plt.ylabel('Predicted Value')
plt.title('Smoothed Predicted Values of All Models vs Target Data')
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
plt.yscale('symlog', linthresh=0.01)

# Annotate the R² scores on the bars
for i, (key, value) in enumerate(r2_scores.items()):
    plt.text(i, value, f"{value:.7f}", ha='center', va='bottom')

# Save the R² scores bar chart as a PNG file
plt.savefig('r2_scores_barchart.png', format='png', dpi=300, bbox_inches='tight')
plt.show()