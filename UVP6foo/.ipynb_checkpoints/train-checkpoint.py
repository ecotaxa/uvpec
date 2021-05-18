import xgboost as xgb

# train 'optimal' model
def train(best_tree_number, dtrain, params):
    
    bst = xgb.train(params, dtrain, num_boost_round = best_tree_number)
    
    return(bst)