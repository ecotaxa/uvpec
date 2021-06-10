#
# UVP6 model training
#

import argparse
import sys
import logging
import os
import uvpec
import pandas as pd
import shutil
from key_generator.key_generator import generate
from zipfile import ZipFile

def main():

    # Parse command line arguments ----
    parser = argparse.ArgumentParser(
        prog="uvpec",
        description="Train UVP models"
    )

    parser.add_argument("path", type=str, nargs=1,
                        help="path to config file")

    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="print debug messages.")

    args = parser.parse_args()

    # Read configuration file ----
    config_file = args.path[0]
    cfg = uvpec.read_config(config_file)
    
    # Read output directory
    output_dir = cfg['io']['output_dir'] 

    # Read features ID
    features_ID = cfg['io']['features_ID']
    
    # Generate unique key to have a unique identification (ID)
    key = generate(1, min_atom_len = 8, max_atom_len = 8).get_key() # unique key of 8 characters

    # Check if output directory exists
    if not os.path.exists(output_dir):
        print("Output directory does not exist. Creating it.")
        os.makedirs(output_dir, exist_ok=True) # make directory
        shutil.copy(config_file, output_dir) # copy config file in that directory
        os.rename(os.path.join(output_dir,'config.yaml'), os.path.join(output_dir, 'config_'+key+'.yaml')) # rename config file with the key 
    else:
        print("Output directory already exists")
        shutil.copy(config_file, output_dir) # copy config file in that directory
        os.rename(os.path.join(output_dir,'config.yaml'), os.path.join(output_dir, 'config_'+key+'.yaml')) # rename config file with the key 

    # Setup logging ----
    log = uvpec.log(output_dir, debug=args.debug)
    log.debug("we're debugging !")

    ### Extract features (pipeline - step 1)
    path_to_subfolders = cfg['io']['images_dir']

    # Zip image folders in the output folder (source: https://www.geeksforgeeks.org/working-zip-files-python/)
    print('Zip image folders')
    file_paths = uvpec.custom.get_all_file_paths(path_to_subfolders)

    # writing files to a zipfile
    if(os.path.isfile(os.path.join(output_dir, features_ID+'_images.zip')) == True):
        print('Images have already been zipped !')
    else:
        with ZipFile(os.path.join(output_dir, features_ID+'_images.zip'),'w') as zip:
            # writing each file one by one
            for file in file_paths:
                zip.write(file)
  
    print('All files zipped successfully!')

    # check if features file exists 
    if(os.path.isfile(os.path.join(output_dir, features_ID+'.feather')) == True):
        print('All features have already been extracted...Loading data')
        dataset = pd.read_feather(os.path.join(output_dir, features_ID+'.feather'))  
    else:
        print("Features file does not exist...Extracting features...")
        # extraction of features 
        # note: We will loose some images that are empty (full black images) so some messages will be printed in the console, this is a normal behaviour
        dataset = uvpec.extract_features(path_to_subfolders)
        # save dataset
        dataset.to_feather(os.path.join(output_dir, features_ID+'.feather'))
        print("We are done with the extraction of features, data have been saved")

    ### Train model (pipeline - step 2)

    # training_data
    df_train = dataset.copy()

    # read xgboost settings
    random_state = cfg['xgboost']['random_state']
    n_jobs = cfg['xgboost']['n_jobs']
    learning_rate = cfg['xgboost']['learning_rate']
    max_depth = cfg['xgboost']['max_depth']
    detritus_subsampling = cfg['xgboost']['detritus_subsampling']
    subsampling_percentage = cfg['xgboost']['subsampling_percentage']
    weight_sensitivity = cfg['xgboost']['weight_sensitivity']
    num_trees_CV = cfg['xgboost']['num_trees_CV']

    # subsample detritus
    if detritus_subsampling:
        df_train = uvpec.sample_detritus(df_train, subsampling_percentage, random_state)
    else:
        None
        
    # Generate class weights 
    class_weights = uvpec.weights(df_train, weight_sensitivity)
        
    # add weights to training set
    weights = df_train[['labels']].replace(to_replace = class_weights, inplace=False)['labels']
    df_train['weights'] = weights

    # do 3-fold cross-validation
    print(df_train.head(n=5)) # just a check
    print(df_train.shape)
    cv_results, xgb_params, config_params, dtrain = uvpec.cross_validation(df_train, num_trees_CV, n_jobs, learning_rate, max_depth, random_state, weight_sensitivity, detritus_subsampling, subsampling_percentage)
    
    # save cv results and compute stats
    best_tree_number = uvpec.save_cv_info(output_dir, cv_results, config_params, key)
    
    # train best model
    best_model = uvpec.train(best_tree_number, dtrain, xgb_params)
    
    # save best model
    uvpec.save_model(best_model, output_dir, key)
    
    # create TAXOCONF file
    MODEL_REF = 'Muvpec_'+key
    uvpec.create_taxoconf(output_dir, class_weights, MODEL_REF, key)

if __name__ == "__main__":
    main()
