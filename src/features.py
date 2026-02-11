from sklearn.preprocessing import OneHotEncoder

def onehotencoder(
        categorical_predictors = list[str],
        drop = 'first',
        handle_unkown = 'ignore'
        ):
    
    return OneHotEncoder(
        categories=categorical_predictors,
        drop=drop,
        handle_unknown=handle_unkown
    )