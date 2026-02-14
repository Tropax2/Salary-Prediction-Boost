# Salary Prediction using Boost 

In this project we use boost to predict the salary of baseball players in the season 1986 and 1987 seasons. The dataset is included in the files
since it comes from the book Introduction to Statistical Learning by James, Witten, Hastie, Tibshirani and Taylor.

## Objective

The objective of this project is to study how both the training and test mean squared error evolve as the learning rate (also known as the shrinkage parameter) changes. Boosting is performed using the `scikit-learn`'s `GradientBoostingRegressor`. To isolate the effect of the learning rate, all other model parameters are kept fixed, and we do not compare against other models.

## Project Structure 

- `src`- source for project architecture, models and evaluations;
- `reports` - detailed report of the results obtained with visual plots. 

## The Dataset 

The dataset includes predictors capturing information about performance records and salaries of baseball players from the 1986 and 1987 seasons. The response variable is the target variable representing the salary of the player. For more information, see `reports_and_results/report.md`.

## Methods 

The `Salary` response variable proved to be trongly right-skewed and so it is log-transformed. Categorical predictors are one-hot-encoded. Since only boosting was to be applied, we didn't see any relevance in standardising the remaining predictors. In order to study how the learning rate affects both the training and test MSE, the most relevant parameters of the method, such as the number of trees, the max depth of each tree and cost-complexity are fixed.

## Results Summary 

 Training MSE follows a decreasing monotonic pattern when we increase the learning rate, resulting in an almost perfect fit of the training data. However, the test MSE didn't exhibit a similar pattern, it follows an inverted U-shape and noticeable fluctuations that are the result of overfitting. In our run, the lowest test MSE was of approximately 0.165.

## How to run 

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\Activate.ps1 # Windows PowerShell
```

### 2) Install Dependencies
```bash 
pip install -r requirements.txt
```

### 3) Run the Project 
```bash
python src/main.py
```

