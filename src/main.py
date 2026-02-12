import data 
import features
from models.boost import boost
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

# This is the main function 
def main():
    # Load the data, transform into a pandas df and remove rows with empty values 
    Hitters = data.csv_to_df('/Users/afonsolopes/Salary-Prediction-Boost/Hitters.csv')

    # log-transform the Salary column 
    Hitters_log_transformed = data.log_transform_salaries(Hitters)
    
    # Separate the predictors and the response 
    predictors = Hitters_log_transformed[["League", "Division", "NewLeague", "AtBat", "Hits", "HmRun", "Runs", "RBI", "Walks", "Years",
    "CAtBat", "CHits", "CHmRun", "CRuns", "CRBI", "CWalks",
    "PutOuts", "Assists", "Errors"]]
    response = Hitters_log_transformed['Salary']

    # Define the categorical predictors to be one-hot-encoded
    categorical_predictors = ["League", "Division", "NewLeague"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = data.make_splits(X = predictors, y = response)

    # One-hot-encode categorical predictors 
    enc = features.onehotencoder(categorical_predictors = categorical_predictors)
    X_train_transformed = enc.fit_transform(X_train)
    X_test_transformed = enc.transform(X_test)
    
    # Perform boost on the training and test set with variable learning_rate with fixed 1000 trees , max_depth of 3 and ccp_alpha of 0
    learning_rates = [0.005, 0.001, 0.02, 0.05, 0.1, 0.2]
    training_mses, test_mses = [], []

    for learning_rate in learning_rates:
        clf = boost(
            learning_rate = learning_rate,
            max_depth = 3,
            n_estimators = 1000,
            ccp_alpha = 0
            )
        clf.fit(X_train_transformed, y_train)

        training_mse = mean_squared_error(y_train, clf.predict(X_train_transformed))
        training_mses.append(training_mse)

        test_mse = mean_squared_error(y_test, clf.predict(X_test_transformed))
        test_mses.append(test_mse)

    
    # Plot of the training mses vs the learning rates 
    plt.figure()
    plt.plot(learning_rates, training_mses)
    plt.xlabel("Learning Rate")
    plt.ylabel("Training MSE")
    plt.title("Training MSE vs Learning Rate")
    plt.grid(True)
    plt.show()

    # Plot of the test mses vs the learning rates 
    plt.figure()
    plt.plot(learning_rates, test_mses)
    plt.xlabel("Learning Rate")
    plt.ylabel("Test MSE")
    plt.title("Test MSE vs Learning Rate")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()