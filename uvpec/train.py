import xgboost as xgb

# train 'optimal' model
def train(best_tree_number, dtrain, xgb_params):

    """
    Function that trains a model based on the DMatrix, the best tree number and the XGBoost parameters given in input.
    Output the model in XGBoost format.
    """
    
    bst = xgb.train(xgb_params, dtrain, num_boost_round = best_tree_number)
    
    return(bst)
