#
# UVP6 model training
#

import argparse
import sys
import logging
import os
import UVP6foo
import pandas as pd
import shutil

def main():

    # Parse command line arguments ----
    parser = argparse.ArgumentParser(
        prog="UVP6foo",
        description="Train UVP models"
    )

    parser.add_argument("path", type=str, nargs=1,
                        help="path to config file")

    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="print debug messages.")

    args = parser.parse_args()

    # Read configuration file ----
    config_file = args.path[0]
    cfg = UVP6foo.read_config(config_file)
    
    # Read output directory
    output_dir = cfg['io']['output_dir'] 
    
    # Check if output directory exists
    if not os.path.exists(output_dir):
        print("Output directory does not exist. Creating it.")
        os.makedirs(output_dir, exist_ok=True) # make directory
        shutil.copy(config_file, output_dir) # copy config file in that directory
    else:
        print("Output directory already exists. Erasing config file with the one provided.")
        shutil.copy(config_file, output_dir)

    # Setup logging ----
    log = UVP6foo.log(output_dir, debug=args.debug)
    log.debug("we're debugging !")

    ### Extract features (pipeline - step 1)
    path_to_subfolders = cfg['io']['images_dir']

    # check if features file exists
    if(os.path.isfile(os.path.join(output_dir, 'features.feather')) == True):
        print('All features have already been extracted...Loading data')
        dataset = pd.read_feather(os.path.join(output_dir, 'features.feather'))
    else:
        print("Features file does not exist...Extracting features...")
        # extraction of features 
        # note: We will loose some images that are empty (full black images)
        dataset = UVP6foo.extract_features(path_to_subfolders)
        # save dataset
        dataset.to_feather(os.path.join(output_dir, 'features.feather'))
        print("We are done with the extraction of features, data have been saved")

    ### Train model (pipeline - step 2)

    # read training_data
    df_train = pd.read_feather(os.path.join(output_dir,'features.feather'))

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
        df_train = UVP6foo.sample_detritus(df_train, subsampling_percentage, random_state)
    else:
        None
        
    # Generate class weights 
    class_weights = UVP6foo.weights(df_train, weight_sensitivity)
        
    # add weights to training set
    weights = df_train[['labels']].replace(to_replace = class_weights, inplace=False)['labels']
    df_train['weights'] = weights
    
    # do 3-fold cross-validation
    print(df_train.head(n=5))
    cv_results, params, dtrain = UVP6foo.cross_validation(output_dir, df_train, num_trees_CV, n_jobs, learning_rate, max_depth, random_state, weight_sensitivity, detritus_subsampling, subsampling_percentage)
    
    # save cv results and compute stats
    best_tree_number = UVP6foo.save_cv_info(output_dir, cv_results, params)
    
    # train best model
    best_model = UVP6foo.train(best_tree_number, dtrain, params)
    
    # save best model
    UVP6foo.save_model(best_model, output_dir, params)
    

if __name__ == "__main__":
    main()
