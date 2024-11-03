import os

import pandas as pd #used to work with python data frames
from sklearn.linear_model import LinearRegression #linear regression model
from sklearn.metrics import  r2_score #scoring for sci kit learn
from sklearn.ensemble import GradientBoostingRegressor


# Path to the specific parquet file in the partition
parquet_file_path = 'train.parquet/partition_id=0/part-0.parquet' #using the 0th partition to train the initial dataset
test_file_path = 'train.parquet/partition_id=1/part-0.parquet' #sample test using 1st of the dataset

#load parquet file with pandas dataframe
df = pd.read_parquet(parquet_file_path) #may need pyarrow extension for this
df = df.fillna(0) #replace all null values with 0


X_train = df.drop('responder_6', axis=1) #remove target value from training set
y_train = df['responder_6'] #y value used to train is selected here

testdf = pd.read_parquet(test_file_path) #read parquet file with pandas
testdf.fillna(0, inplace=True) #replace all null values with 0 for the test set like we did with the first data set
X_test = testdf.drop('responder_6', axis=1) #remove target from test features
y_test = testdf['responder_6'] # target value for test

regressors = {
    'Linear Regression':LinearRegression(),
    'Gradient Boost':GradientBoostingRegressor(random_state=0)
}




regressors_results = []


for regressor_name, regressor in regressors.items():
    regressor.fit(X_train, y_train) #train model using features and target out put
    y_pred = regressor.predict(X_test) #y pred will be the models predictions for y test on the feature list X_text

    # Calculate R2 score
    r2 = r2_score(y_test, y_pred)
    regressors_results.append(regressor_name + ' r2 score: ' + str(r2))
    print("R² Score:", r2)

print("\nResults:")
for i in regressors_results:
    print(i)