import os

import pandas as pd #used to work with python data frames
import polars as pl #used to load parquet files from train and test

# save Kaggle dataset you your local folder and extract


# Path to the specific parquet file in the partition
parquet_file_path = 'train.parquet/partition_id=0/part-0.parquet'

# Load the parquet file using Polars
df = pl.read_parquet(parquet_file_path)

# Display the first few rows of the dataframe
print(df.head())
print(df.row(0)) 