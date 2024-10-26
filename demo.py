import os

import pandas as pd #used to work with python data frames
from sklearn.linear_model import LinearRegression #linear regression model
from sklearn.metrics import  r2_score #scoring for sci kit learn


# Path to the specific parquet file in the partition
parquet_file_path = 'train.parquet/partition_id=0/part-0.parquet' #using the 0th partition to train the initial dataset

#load parquet file with pandas dataframe
df = pd.read_parquet(parquet_file_path) #may need pyarrow extension for this
df = df.fillna(0) #replace all null values with 0


X_train = df.drop('responder_6', axis=1) #remove target value from training set
y_train = df['responder_6'] #y value used to train is selected here

model = LinearRegression() #linear regression model from scikit learn
model.fit(X_train, y_train) #train model using features and target out put


test_file_path = 'train.parquet/partition_id=1/part-0.parquet' #sample test using 1st of the dataset
testdf = pd.read_parquet(test_file_path) #read parquet file with pandas

testdf.fillna(0, inplace=True) #replace all null values with 0 for the test set like we did with the first data set
X_test = testdf.drop('responder_6', axis=1) #remove target from test features
y_test = testdf['responder_6'] # target value for test

y_pred = model.predict(X_test) #y pred will be the models predictions for y test on the feature list X_text

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

print("R² Score:", r2)
print("\nTop 10 Predictions")
i = 0
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted}")
    i = i + 1
    if i > 10:
        break