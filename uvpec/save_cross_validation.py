import os
import pandas as pd
import numpy as np

def save_cv_info(output_dir, cv_model, params, key):

    """
    Function that saves the panda dataframe containing information on the cross-validation (i.e. to see if it overfits with a plot logloss in fonction of the number of trees. 
    It also outputs a txt file containing information on the best tree number (logloss associated to it and also accuracy).
    It returns the best tree number according to the CV.
    """
    
    print('Save CV info and compute best tree number to train the model.')
    print('Configuration parameters :')
    print(params)
    # save cross-validation model
    cv_model.to_feather(os.path.join(output_dir,'inflexion_point_'+key+'.feather'))
    
    # Compute some stats/metrics
    
    # Find optimal tree number (will be used to train the model with the ENTIRE training set, unlike a CV)
    best_tree_number = np.argmin(cv_model['test-mlogloss-mean'])+1
    # some numbers for the CV (best mlogloss and the accuracy related to it) 
    best_logloss = cv_model.iloc[best_tree_number-1]['test-mlogloss-mean']
    accuracy_cv = 1 - cv_model.iloc[best_tree_number-1]['test-merror-mean'] # accuracy = 1 - merror (multiclass error)
    
    # write stats/metrics
    with open(os.path.join(output_dir,'cv_results_'+key+'.txt'), 'w') as f:
        f.write("Best tree number for the cross-validation is %d\r\n" % best_tree_number)
        f.write("Best mlogloss is %f\r\n" % best_logloss)
        f.write("Accuracy for this tree number %f\r\n" % accuracy_cv)

    f.close()  
    
    return(best_tree_number)
