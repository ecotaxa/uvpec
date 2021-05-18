import UVP6foo
from UVP6foo.custom import int_to_label, label_to_int
import xgboost as xgb
import numpy as np
import pandas as pd

def cv_train(project_dir, dataset, weights, n_jobs, learning_rate, max_depth, num_class, random_state):
    """
    Function that first does a 3-fold cross-validation to find the best number of trees then train the model based on that 'best' number.
    It also saves the model in xgboost and binary formats.
    """
    
    # create a dict() to convert labels to int (necessary for xgboost training that only works with int and not characters)
    class_counts = dataset.groupby('labels').size()
    dico_label = {} 
    for i, idx in enumerate(class_counts.items()):
        dico_label.update({idx[0]:i})
    
    # construction of the DMatrix for XGBoost training
    y_train = label_to_int(dico_label, dataset['labels'])
    df_train = dataset.iloc[:,0:len(dataset.columns)-3]
    weights =  df_train['weights']
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
    bst.to_feather(os.path.join(project_dir,'inflexion_point_'+str(learning_rate)+'_'+str(max_depth)+'_'+str(use_weights)+'_'+str(weight_sensitivity)+'_'+str(detritus_subsampling)+'_'+str(subsampling_percentage)+'.feather'))
    
    # Find optimal tree number (will be used to train the model with the ENTIRE training set, unlike a CV)
    best_tree_number = np.argmin(bst['test-mlogloss-mean'])+1
    
    # some numbers for the CV (best mlogloss and the accuracy related to it) 
    best_logloss = bst.iloc[best_tree_number-1]['test-mlogloss-mean']
    accuracy_cv = 1 - bst.iloc[best_tree_number-1]['test-merror-mean'] # accuracy = 1 - merror (multiclass error)
    
    # write the 'best' metrics of the cross-validation in a text file
    with open(os.path.join(project_dir,'cv_xgboost_best_param_'+str(learning_rate)+'_'+str(max_depth)+'_'+str(use_weights)+'_'+str(weight_sensitivity)+'_'+str(detritus_subsampling)+'_'+str(subsampling_percentage)+'.txt'), 'w') as f:
        f.write("Best tree number for the cross-validation is %d\r\n" % best_tree_number)
        f.write("Best mlogloss is %f\r\n" % best_logloss)
        f.write("Accuracy for this tree number %f\r\n" % accuracy_cv)
    
    f.close()    
    
    # train 'optimal' model
    bst = xgb.train(params, dtrain, num_boost_round = best_tree_number)
    
    # save model in xgboost format
    print('Saving model in xgboost format and text format.')
    bst.save_model(os.path.join(project_dir,'xgboost_'+str(learning_rate)+'_'+str(max_depth)+'_'+str(use_weights)+'_'+str(weight_sensitivity)+'_'+str(detritus_subsampling)+'_'+str(subsampling_percentage)+'.model'))
    
    # to be done (WISIP)
    # dump model to binary format