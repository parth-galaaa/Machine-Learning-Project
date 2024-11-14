import os
import pandas as pd #used to work with python data frames
from sklearn.linear_model import LinearRegression #linear regression model
from sklearn.metrics import  r2_score #scoring for sci kit learn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Input, Dropout, MaxPooling1D, LSTM
from tensorflow.keras.models import Sequential
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import csv
import time
# Path to the specific parquet file in the partition
parquet_file_path = 'train.parquet/partition_id=0/part-0.parquet' #using the 0th partition to train the initial dataset

#load parquet file with pandas dataframe
df = pd.read_parquet(parquet_file_path) #may need pyarrow extension for this
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the DataFrame and transform it
df.dropna(axis=1, how='all', inplace=True)
df[df.columns] = imputer.fit_transform(df[df.columns])

# Split the data into training and testing sets
split_index = int(len(df) * 0.2)
df = df[:split_index]
print(len(df))
split_index = int(len(df) * 0.8)
train_df = df[:split_index]  # Top 80% for training
test_df = df[split_index:]   # Last 20% for testing
scaler = StandardScaler()

# Shift the responder columns up by 1
lag = 1
for responder in range(9):
    if responder != 6:
        col_name = f'responder_{responder}'
        train_df.loc[:, col_name] = train_df[col_name].shift(lag)
        test_df.loc[:, col_name] = test_df[col_name].shift(lag)
print(test_df)
train_df = train_df.dropna()
test_df = test_df.dropna()

# Split the data into features and target
# Training data
X_train = train_df.drop('responder_6', axis=1) #remove target value from training set
y_train = train_df['responder_6'] #y value used to train is selected here

# Testing data
X_test = test_df.drop('responder_6', axis=1) #remove target from test features
y_test = test_df['responder_6'] # target value for test

# Scale the data
X_train  = scaler.fit_transform(X_train )
X_test = scaler.transform(X_test)
num_columns = X_train.shape[1]

# Define the regressors
def CNN():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(num_columns, 1)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def kerasLSTM():
    model = Sequential()
    model.add(LSTM(4, activation='relu',input_shape=(num_columns, 1)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def Hybrid():
    model = Sequential()
    model.add(Conv1D(8, kernel_size=3, activation='relu', input_shape=(num_columns, 1)))
    model.add(LSTM(4, activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Dictionary of regressors
regressors = {
    'Linear Regression': LinearRegression(),
    'Gradient Boost':HistGradientBoostingRegressor(random_state=0),
    'Random Forest Regressor':RandomForestRegressor(
    n_estimators=50, #reduced trees to 50
    max_depth=10,    #Set limit on depth of each tree
    n_jobs=-1,        # Use all CPU cores
    random_state=42 ),
    'CNN': CNN(),
    'LSTM':kerasLSTM(),
    'MLP Regressor': MLPRegressor(hidden_layer_sizes=(100,), max_iter=200, random_state=42),
    'Hybrid':Hybrid()
}

# Dictionary to hold results
regressors_results = {}

# Add target data to results
regressors_results['TARGET DATA'] = y_test

# List to hold timing results for CSV
timing_results = []

# Train and predict for each regressor
for regressor_name, regressor in regressors.items():
    print(f"Starting training for {regressor_name}...")
    
    # Record start time
    start_time = time.time()
    
    # Train model
    if regressor_name in ('CNN', 'Hybrid', 'LSTM'):
        regressor.fit(X_train, y_train, epochs=5, batch_size=16, verbose=2)
    else:
        regressor.fit(X_train, y_train)
        
    # Record end time
    end_time = time.time()
    
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Training for {regressor_name} completed in {elapsed_time:.2f} seconds.")
    
    # Append results to list
    timing_results.append([regressor_name, elapsed_time])
    
    # Predict and process results
    y_pred = regressor.predict(X_test)
    
    if regressor_name in ('CNN', 'Hybrid', 'LSTM'):
        y_pred = list(y_pred)
    
    # Calculate R2 score
    r2 = r2_score(y_test, y_pred)
    regressors_results[regressor_name] = y_pred
    print(f"{regressor_name} R2 score: {r2:.4f}\n")

# Write timing results to CSV
with open('training_times.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Regressor', 'Training Time (seconds)'])
    writer.writerows(timing_results)

print("Training times saved to 'training_times.csv'")

# Convert results to DataFrame
results_df = pd.DataFrame(regressors_results)

# Save DataFrame to CSV
results_df.to_csv('regressors_results.csv', index=False)