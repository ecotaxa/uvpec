#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import uvp6lib as uvp6
import xgboost as xgb
import read_settings
import Florian_custom as flo

# read global settings
global_settings = read_settings.check_global()
dataFolder = global_settings['pipeline_folder']
experimentFolder = global_settings['experiment_folder']

# read training_data
df_train = pd.read_feather(os.path.join(os.getcwd(),dataFolder,experimentFolder,'features.feather'))

# read xgboost settings
xgboost_settings = read_settings.check_xgboost()

# settings
random_state = global_settings['random_state']
n_jobs = xgboost_settings['n_jobs']
use_weights = xgboost_settings['use_weights']
learning_rate = xgboost_settings['learning_rate']
max_depth = xgboost_settings['max_depth']
detritus_subsampling = xgboost_settings['detritus_subsampling']
subsampling_percentage = xgboost_settings['subsampling_percentage']
weight_sensitivity = xgboost_settings['weight_sensitivity']
num_trees_CV = xgboost_settings['num_trees_CV']

def sample_detritus(dataset, subsampling_percentage, random_state):
    """
    Function that randomly selects detritus based on a user-specified percentage
    """
    
    detritus = dataset[dataset['labels'] == 'detritus']
    other_classes = dataset[dataset['labels'] != 'detritus']
    detritus = detritus.sample(frac = subsampling_percentage/100, replace = False, random_state = random_state)
    new_dataset = pd.concat([detritus, other_classes])
    new_dataset = new_dataset.reset_index(inplace=False, drop = True)
    
    return(new_dataset)

# subsample detritus
if detritus_subsampling:
    df_train = sample_detritus(df_train, subsampling_percentage, random_state)
else:
    None

# source : https://github.com/ThelmaPana/plankton_classif_benchmark/blob/main/train_rf.py
# Generate class weights
import math
class_counts = df_train.groupby('labels').size()
count_max = 0
class_weights = {}
if use_weights:
    count_max = np.max(class_counts)
    for i,idx in enumerate(class_counts.items()):
        class_weights.update({idx[0] : (count_max / idx[1])**weight_sensitivity})
else:
    for idx in class_counts.items():
        class_weights.update({idx[0] : 1.0})
        
# add weights to training set
weights = df_train[['labels']].replace(to_replace = class_weights, inplace=False)['labels']
df_train['weights'] = weights

# create a dict() to convert labels to int (for xgboost)
dico_label = {} 
for i, idx in enumerate(class_counts.items()):
    dico_label.update({idx[0]:i})

# construction of the DMatrix for XGBoost training
y_train = flo.label_to_int(dico_label, df_train['labels'])
df_train = df_train.iloc[:,0:len(df_train.columns)-3]
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
print('Cross-validation is done. Model is now trained with the best tree number found.')

# save bst to look later at the inflexion point
path_to_subfolders = os.path.join(os.getcwd(), dataFolder, experimentFolder)
bst.to_feather(os.path.join(path_to_subfolders,'inflexion_point_'+str(learning_rate)+'_'+str(max_depth)+'_'+str(use_weights)+'_'+str(weight_sensitivity)+'_'+str(detritus_subsampling)+'_'+str(subsampling_percentage)+'.feather'))

# Find optimal tree number (will be used to train the model with the ENTIRE training set, unlike a CV)
best_tree_number = np.argmin(bst['test-mlogloss-mean'])+1

# some numbers for the CV (best mlogloss and the accuracy related to it) 
best_logloss = bst.iloc[best_tree_number-1]['test-mlogloss-mean']
accuracy_cv = 1 - bst.iloc[best_tree_number-1]['test-merror-mean'] # accuracy = 1 - merror (multiclass error)

# write the 'best' metrics of the cross-validation in a text file
with open(os.path.join(path_to_subfolders,'cv_xgboost_best_param_'+str(learning_rate)+'_'+str(max_depth)+'_'+str(use_weights)+'_'+str(weight_sensitivity)+'_'+str(detritus_subsampling)+'_'+str(subsampling_percentage)+'.txt'), 'w') as f:
    f.write("Best tree number for the cross-validation is %d\r\n" % best_tree_number)
    f.write("Best mlogloss is %f\r\n" % best_logloss)
    f.write("Accuracy for this tree number %f\r\n" % accuracy_cv)

f.close()    

# train 'optimal' model
bst = xgb.train(params, dtrain, num_boost_round = best_tree_number)

# save model in xgboost format
print('Saving model in xgboost format and text format.')
bst.save_model(os.path.join(path_to_subfolders,'xgboost_'+str(learning_rate)+'_'+str(max_depth)+'_'+str(use_weights)+'_'+str(weight_sensitivity)+'_'+str(detritus_subsampling)+'_'+str(subsampling_percentage)+'.model'))

# dump model to a text file
bst.feature_names = None # clean model feature names before exporting
bst.dump_model(os.path.join(path_to_subfolders,'xgboost_'+str(learning_rate)+'_'+str(max_depth)+'_'+str(use_weights)+'_'+str(weight_sensitivity)+'_'+str(detritus_subsampling)+'_'+str(subsampling_percentage)+'_model.txt'))

# to be done (WISIP)
# dump model to binary format
