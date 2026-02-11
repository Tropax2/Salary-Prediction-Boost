import data 


# This is the main function 
def main():
    # Load the data, transform into a pandas df and remove rows with empty values 
    Hitters = data.csv_to_df('/Users/afonsolopes/Salary-Prediction-Boost/Hitters.csv')

    # log-transform the Salary column 
    Hitters_log_transformed = data.log_transform_salaries(Hitters)
    
    # Separate the predictors 
    categorical_predictors = ["League", "Division", "NewLeague"]
    numerical_predictors = [
    "AtBat", "Hits", "HmRun", "Runs", "RBI", "Walks", "Years",
    "CAtBat", "CHits", "CHmRun", "CRuns", "CRBI", "CWalks",
    "PutOuts", "Assists", "Errors"
    ]   
    response = Hitters_log_transformed['Salary']
    


if __name__ == '__main__':
    main()