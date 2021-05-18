import os

def save_model(xgb_model, output_dir, params):

    """
    Function that saves the model in the output directory in XGBoost format and also in a text file.
    """
    
    # save model in xgboost format
    print('Saving model in xgboost format and text format.')
    xgb_model.save_model(os.path.join(output_dir,'xgboost_'+str(params['learning_rate'])+'_'+str(params['max_depth'])+'_'+str(params['weight_sensitivity'])+'_'+str(params['detritus_subsampling'])+'_'+str(params['subsampling_percentage'])+'.model'))
    
    # dump model to a text file
    xgb_model.feature_names = None # clean model feature names before exporting
    xgb_model.dump_model(os.path.join(output_dir,'xgboost_'+str(params['learning_rate'])+'_'+str(params['max_depth'])+'_'+str(params['weight_sensitivity'])+'_'+str(params['detritus_subsampling'])+'_'+str(params['subsampling_percentage'])+'_model.txt'))

    # to be done (WISIP)
    # dump model to binary format
    
