import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#Load the dataset
file_path = r'C:\Users\Ibtas\School\FALL 2024\COMP 4730\Project\jane-street\train.parquet\partition_id=0'
data = pd.read_parquet(file_path)

#10% of data
data = data.sample(frac=0.1, random_state=42)

# Check for missing data in the entire dataset before any imputation
print("Missing values in the dataset before imputation:")
print(data.isnull().sum())

# Investigate columns with missing data in the feature set (feature_00 to feature_79)
feature_columns = [f'feature_{i:02d}' for i in range(79)]
print("Missing values in feature columns before filling:")
print(data[feature_columns].isnull().sum())

# Identify and drop columns with all missing values
all_nan_columns = data[feature_columns].isnull().sum()[data[feature_columns].isnull().sum() == len(data)]
print("Columns with all missing values in feature columns:")
print(all_nan_columns)
data.drop(columns=all_nan_columns.index, inplace=True)

# Check again for missing values after dropping empty columns
print("Missing values in dataset after dropping all-NaN columns:")
print(data.isnull().sum())

#check for missing data, fill with mean
data.fillna(data.mean(), inplace=True)

# Select remaining feature columns (now without dropped columns)
feature_columns = [col for col in data.columns if col.startswith('feature_')]
x = data[feature_columns]
y = data['responder_0']

#Feature Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Check for Nan values in the features and target columns before splitting
#print("Nan values in feature columns before split:")
#print(x.isnull().sum())
#print("Nan values in target column before split:")
#print(y.isnull().sum())

#Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Using mean strategy to fill NaNs that appear after split
imputer = SimpleImputer(strategy='mean')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

#Fill NaN values in y_train 
y_train.fillna(y_train.mean(), inplace=True)

#intitalize the Linear Regression Model
model = LinearRegression()
#Train the model using the training data
model.fit(x_train, y_train)

#predict the target values for the test set
y_pred = model.predict(x_test)
#Evaluate the model using mean squared error and R^2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#print the metrics
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

#Check shape of resulting splits
print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

#Information of the dataset
print(data.info())

#Describe the dataset
print(data.describe())

#predict the target values for the test set
y_pred = model.predict(x_test)
print("Predicted values:", y_pred)  # Add this line to see predictions
