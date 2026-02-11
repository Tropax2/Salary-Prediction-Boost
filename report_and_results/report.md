## The Dataset 

The dataset contains information about performance records and salaries of baseball players from the 1986 and 1987 seasons. The columns of the dataset are the following:

- **AtBat**: Number of times at bat in 1986.
- **Hits**: Number of hits in 1986.
- **HmRun**: Number of home runs in 1986.
- **Runs**: Number of runs in 1986.
- **RBI**: Number of runs batted in 1986.
- **Walks**: Number of walks in 1986.
- **Years**: Number of years in the major leagues.
- **CAtBat**: Number of times at bat in the career.
- **CHits**: Number of hits in the career.
- **CHmRun**: Number of home runs in the career.
- **CRuns**: Number of runs in the career.
- **CRBI**: Number of runs batted in the career.
- **CWalks**: Number of walks in the career.
- **League**: A factor with levels A and N indicating player's league at the end of 1986.
- **Division**: A factor with levels E and W indicating player's division at the end of 1986.
- **PutOuts**: Number of put outs in 1986.
- **Assists**: Number of assists in 1986.
- **Errors**: Number of errors in 1986.
- **Salary**: 1987 annual salary on opening day in thousands of dollars.

The target variable is the `Salary` variable, while the others are predictor variables.

## Data Processing 

The CSV is loaded into a pandas DataFrame and rows with missing values are dropped. The `Salary` column is log-transformed due to the fact that 
such column is strongly right-skewed since there could be players that earn much more than the remaining players. In fact, the skewness of `Salary` is of 1.59, while of log-transformed `Salary` is of -0.18. Categorical predictors are one-hot encoded using `OneHotEncoder`. 

