import UVP6foo
from UVP6foo.custom import int_to_label, label_to_int
import xgboost as xgb
import numpy as np
import pandas as pd
import os

def cv_train(project_dir, dataset, num_trees_CV, n_jobs, learning_rate, max_depth, random_state, weight_sensitivity, detritus_subsampling, subsampling_percentage):
    """
    Function that first does a 3-fold cross-validation (CV) to find the best number of trees then train the final model based on that number.
    It also saves the model in xgboost and binary formats. To check the CV, 2 additional files (cv_xgboost_best_param_XXX and inflexion_point_XXX) will also be saved.
    """

    # create a dictionary to convert labels to int (necessary for training with XGBoost because it only works with integers and not characters)
    class_counts = dataset.groupby('labels').size()
    dico_label = {}
    for i, idx in enumerate(class_counts.items()):
        dico_label.update({idx[0]:i})

    # construction of the DMatrix for XGBoost training
    y_train = label_to_int(dico_label, dataset['labels'])
    df_train = dataset.drop(columns = ['labels', 'weights']) # drop two non-features columns
    weights =  dataset['weights']
    dtrain = xgb.DMatrix(df_train, label=y_train, weight = np.array(weights))

    # parameters for the cross-validation (CV)
    num_class = len(np.unique(y_train))

    params = {'nthread':n_jobs, 'eta':learning_rate, 'max_depth':max_depth, 'subsample':0.75,
             'tree_method':'hist', 'objective':'multi:softprob',
             'eval_metric':['mlogloss','merror'], 'num_class':num_class,
             'seed':random_state}

    num_boost_round = num_trees_CV

    # CV
    print('Starting the cross-validation with '+str(num_boost_round)+' trees.')
    bst = xgb.cv(params, dtrain, num_boost_round = num_boost_round, nfold = 3, stratified = True,
                 as_pandas = True) # I assume that folds is not useful if yours are using nfold and stratified..
    print('Cross-validation is done.')
    
    return(bst, params, dtrain)