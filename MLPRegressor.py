# Import necessary libraries
from google.colab import files
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Path to the specific parquet file in the partition
parquet_file_path = 'part-0.parquet'

# Load the parquet file
df = pd.read_parquet(parquet_file_path)

# create feature_columns based on the column names
feature_columns = [f'feature_{i:02d}' for i in range(79)]
# only keep columns that exist in the DataFrame
feature_columns = [col for col in feature_columns if col in df.columns]
print("Valid feature columns:", feature_columns)

# Check for missing values in the dataset before imputation
print("Missing values in the dataset before imputation:")
print(df.isnull().sum())

# Handle missing data in features (X) using mean imputation
imputer = SimpleImputer(strategy='mean')
X = df[feature_columns]  # Features
X = imputer.fit_transform(X)  # Impute missing values in X

# Handle missing data in the target (y)
y = df['responder_0']  # Target
y.fillna(y.mean(), inplace=True)  # Impute missing values in y

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the MLP model
mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=200, random_state=42)
mlp.fit(X_train, y_train)

# Predict the target values for the test set
y_pred = mlp.predict(X_test)

# Evaluate the model using Mean Squared Error and R² Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Check shape of resulting splits
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

#Information of the dataset
print(df.info())
#Describe the dataset
print(df.describe())

#predict the target values for the test set
print("Predicted values (y_pred):")
print(y_pred)
