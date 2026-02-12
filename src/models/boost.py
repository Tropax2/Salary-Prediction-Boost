from sklearn.ensemble import GradientBoostingRegressor 

def boost(
        learning_rate,
        n_estimators, 
        max_depth,
        ccp_alpha
):
    return GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        ccp_alpha=ccp_alpha
    )