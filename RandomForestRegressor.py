import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#Load the dataset
file_path = r'C:\Users\Ibtas\School\FALL 2024\COMP 4730\Project\jane-street\train.parquet\partition_id=0'
data = pd.read_parquet(file_path).sample(frac=0.1, random_state=42) #10% of data

#Missing values and scale features
feature_columns = [f'feature_{i:02d}' for i in range(79)]
all_nan_columns = data[feature_columns].isnull().sum()[data[feature_columns].isnull().sum() == len(data)]
data.drop(columns=all_nan_columns.index, inplace=True)

#NaNs, fill with mean
data.fillna(data.mean(), inplace=True)

# Select remaining feature columns (now without dropped columns)
feature_columns = [col for col in data.columns if col.startswith('feature_')]
x = data[feature_columns]
y = data['responder_0']

#Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scaler features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

rf_model = RandomForestRegressor(
    n_estimators=50, #reduced trees to 50
    max_depth=10,    #Set limit on depth of each tree
    n_jobs=-1,        # Use all CPU cores
    random_state=42 
    )
rf_model.fit(x_train, y_train)

#Predict and evaluate 
y_pred = rf_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Model")
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
print("Predicted Values:", y_pred)