import os
from cython_uvp6 import py_convert_txt_model_to_binary_model

def save_model(xgb_model, output_dir, key, n_categories):

    """
    Function that saves the model in the output directory in XGBoost format and also in a text file.
    """
    
    # save model in xgboost format
    print('Saving model in xgboost format and text format.')
    xgb_model.save_model(os.path.join(output_dir,'Muvpec_'+key+'.model'))
    
    # dump model to a text file
    xgb_model.feature_names = None # clean model feature names before exporting
    xgb_model.dump_model(os.path.join(output_dir,'Muvpec_'+key+'.txt'))

    # convert txt model to binary model
    txt_model_bytes = bytes(os.path.join(output_dir,'Muvpec_'+key+'.txt'), encoding ='utf-8') # need to convert string to bytes
    py_convert_txt_model_to_binary_model(n_categories, txt_model_bytes)

    ## add number of categories (i.e. taxa) in the model.txt file (i.e. Muvpec_key.txt)
    #with open(os.path.join(output_dir,'Muvpec_'+key+'.txt'), 'a') as f:
    #     f.write('categories_number='+str(n_categories)+'\n')
