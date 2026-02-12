from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def onehotencoder(
        categorical_predictors: list[str]
        ):
    enc = OneHotEncoder(
            drop = "first",
            handle_unknown = "ignore"
    )

    preprocessor = ColumnTransformer(
        transformers = [
            ("cat", enc, categorical_predictors)
        ],
        remainder="passthrough" 
    )
    return preprocessor