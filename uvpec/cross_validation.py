from uvpec.custom import label_to_int
import xgboost as xgb
import numpy as np
import pandas as pd
import os

def cross_validation(dataset, num_trees_CV, n_jobs, learning_rate, max_depth, random_state, weight_sensitivity, detritus_subsampling, subsampling_percentage):
    """
    Function that does a 3-fold cross-validation (CV) to find the best number of trees for the final training of the UVP6 model.
    It returns the 'CV-model' (not a model per se), a dict() for the parameters used for XGBoost and, a dict() for the parameters
    entered by the user in the configuration file and a DMatrix with all data necessary for the training of the model with XGBoost.
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

    xgb_params = {'nthread':n_jobs, 'eta':learning_rate, 'max_depth':max_depth, 'subsample':0.75,
             'tree_method':'hist', 'objective':'multi:softprob',
             'eval_metric':['mlogloss','merror'], 'num_class':num_class,
             'seed':random_state}

    num_boost_round = num_trees_CV

    # CV
    print('Starting the cross-validation with '+str(num_boost_round)+' trees.')
    bst = xgb.cv(xgb_params, dtrain, num_boost_round = num_boost_round, nfold = 3, stratified = True,
                 as_pandas = True) # I assume that folds is not useful if yours are using nfold and stratified..
    print('Cross-validation is done.')
    
    config_params = {'learning_rate':learning_rate, 'max_depth':max_depth, 'weight_sensitivity':weight_sensitivity, 'detritus_subsampling':detritus_subsampling,'subsampling_percentage':subsampling_percentage}

    return(bst, xgb_params, config_params, dtrain)
