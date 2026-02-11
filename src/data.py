import pandas as pd 
import scipy as sp
import numpy as np
from sklearn.model_selection import train_test_split

# Transform the csv file into a pandas df and remove rows with empty values 
def csv_to_df(csv_path : str) -> pd.DataFrame:
    Hitters = pd.read_csv(csv_path)
    Hitters = Hitters.dropna()

    return Hitters 

# Compute the skewness of Salary
#print(sp.stats.skew(csv_to_df('path')['Salary']))

# log transform the Salary column
def log_transform_salaries(df : pd.DataFrame):
    df['Salary'] = df['Salary'].apply(np.log)

    return df

# Split the data into a training and test set 
def make_splits(
        X, 
        y,
        test_size = 0.3,
        random_state = 42,
):
    return train_test_split(X, 
                            y, 
                            test_size=test_size, 
                            random_state=random_state, 
                            shuffle=True
                            )

    