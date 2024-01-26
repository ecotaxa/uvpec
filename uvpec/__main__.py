#
# UVP6 model training
#

import argparse
import sys
import logging
import os
import uvpec
import pandas as pd
import numpy as np
import shutil
from key_generator.key_generator import generate
from zipfile import ZipFile

def main():

    # Parse command line arguments ----
    parser = argparse.ArgumentParser(
        prog="uvpec",
        description="Train UVP6 models"
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
    
    # read train/test images dir
    path_to_training_subfolders = cfg['io']['train_images_dir']
    path_to_test_subfolders = cfg['io']['test_images_dir']

    # Read training and test features file (if they do not exist, they will be created)
    training_features = cfg['io']['training_features_file']
    test_features = cfg['io']['test_features_file']

    # read xgboost model (can be a dummy filepath if there is no evaluation)
    xgb_model = cfg['io']['model']

    # read objid_threshold file (not compulsory, using a constant threshold is also allowed)
    objid_threshold_file = cfg['io']['objid_threshold_file']

    # read instrument settings
    pixel_threshold = cfg['instrument']['uvp_pixel_threshold']
    use_objid_threshold_file = cfg['instrument']['use_objid_threshold_file']

    # read xgboost settings
    random_state = cfg['xgboost']['random_state']
    n_jobs = cfg['xgboost']['n_jobs']
    learning_rate = cfg['xgboost']['learning_rate']
    max_depth = cfg['xgboost']['max_depth']
    detritus_subsampling = cfg['xgboost']['detritus_subsampling']
    subsampling_percentage = cfg['xgboost']['subsampling_percentage']
    weight_sensitivity = cfg['xgboost']['weight_sensitivity'] 
    num_trees_CV = cfg['xgboost']['num_trees_CV']

    # option to use C instead of python to extract the features
    use_C = cfg['language']['use_C']

    # read process
    train_only = cfg['process']['train_only']
    evaluate_only = cfg['process']['evaluate_only']

    # Generate unique key to have a unique identification (ID)
    key = generate(1, min_atom_len = 8, max_atom_len = 8).get_key() # unique key of 8 characters
    
    # print error message if user wants both an evaluation only and a training only
    if (evaluate_only and train_only):
        print('It seems like you did not fill the configuration file correctly.')
        print('If you want to train a model and evaluate it, choose evaluation_only: false and train_only: false')
        print('If you only want an evaluation, choose evaluation_only: true and train_only: false')
        print('If you only want to train a model, without the evaluation, choose evaluation_only: false and train_only: true')
        print('Please modify the configuration file and run it again')
        sys.exit(0)

    if evaluate_only:
        print('no training, model evaluation only')

        # Check if output directory exists and create it if it does not exist
        if not os.path.exists(output_dir):
            print("Output directory does not exist. Creating it.")
            os.makedirs(output_dir, exist_ok=True) # make directory
            shutil.copy(config_file, output_dir) # copy config file in that directory
            os.rename(os.path.join(output_dir,'config.yaml'), os.path.join(output_dir, 'config_'+key+'.yaml')) # rename config file with the key 
        else:
            print("Output directory already exists")
            shutil.copy(config_file, output_dir) # copy config file in that directory
            os.rename(os.path.join(output_dir,'config.yaml'), os.path.join(output_dir, 'config_'+key+'.yaml')) # rename config file with the key 
        
        # extract test features
        if(os.path.isfile(os.path.join(output_dir, test_features+'.feather')) == True):
            print('Test features have already been extracted...Loading data')
            dataset_test = pd.read_feather(os.path.join(output_dir, test_features+'.feather'))  
            dico_id_test = np.load(os.path.join(output_dir, 'dico_id_test.npy'), allow_pickle=True) # read numpy file
            dico_id_test = dict(enumerate(dico_id_test.flatten(), 1)) # convert numpy ndarray to dict
            dico_id_test = dico_id_test[1] # get the right format for an easy use
        else:
            print("Test features file does not exist...Extracting features...")
            # extraction of features 
            dataset_test, dico_id_test = uvpec.extract_features(path_to_test_subfolders, pixel_threshold, objid_threshold_file, use_objid_threshold_file, use_C)
            # save dataset
            dataset_test.to_feather(os.path.join(output_dir, test_features+'.feather'))
            # save dico_id
            np.save(os.path.join(output_dir,'dico_id_test.npy'), dico_id_test)
            print("We are done with the extraction of test features, data have been saved")    

        uvpec.evaluate_model(n_jobs, dataset_test, xgb_model,'toto', False, output_dir, use_C) # toto because we don't use the inflexion file in the evaluation process only
        sys.exit(0) # evaluation only, we stop here

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
    # Zip image folders in the output folder (source: https://www.geeksforgeeks.org/working-zip-files-python/)
    print('Zip training image folders')

    # writing files to a zipfile
    if(os.path.isfile(os.path.join(output_dir, training_features+'_images.zip')) == True):
        print('Images have already been zipped !')
    else:
        print('Images are being zipped...')
        file_paths = uvpec.custom.get_all_file_paths(path_to_training_subfolders)
        with ZipFile(os.path.join(output_dir, training_features+'_images.zip'),'w') as zip:
            # writing each file one by one
            for file in file_paths:
                zip.write(file)
        print('All files zipped successfully!')   

    # check if features file exists 
    if(os.path.isfile(os.path.join(output_dir, training_features+'.feather')) == True):
        print('Training features have already been extracted...Loading data')
        dataset = pd.read_feather(os.path.join(output_dir, training_features+'.feather'))  
        dico_id = np.load(os.path.join(output_dir, 'dico_id_train.npy'), allow_pickle=True) # read numpy file
        dico_id = dict(enumerate(dico_id.flatten(), 1)) # convert numpy ndarray to dict
        dico_id = dico_id[1] # get the right format for an easy use
    else:
        print("Training features file does not exist...Extracting features...")
        # extraction of features 
        # note: We will loose some images that are empty (full black images) so some messages will be printed in the console, this is a normal behaviour
        dataset, dico_id = uvpec.extract_features(path_to_training_subfolders, pixel_threshold, objid_threshold_file, use_objid_threshold_file, use_C)
        # save dataset
        dataset.to_feather(os.path.join(output_dir, training_features+'.feather'))
        # save dico_id
        np.save(os.path.join(output_dir,'dico_id_train.npy'), dico_id)
        print("We are done with the extraction of training features, data have been saved")

    ### Train model (pipeline - step 2)

    # training_data
    df_train = dataset.copy()

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
    cv_results, xgb_params, config_params, dtrain = uvpec.cross_validation(df_train, num_trees_CV, n_jobs, learning_rate, max_depth, random_state, weight_sensitivity, detritus_subsampling, subsampling_percentage)
    
    # save cv results and compute stats
    best_tree_number = uvpec.save_cv_info(output_dir, cv_results, config_params, key)
    
    # train best model
    best_model = uvpec.train(best_tree_number, dtrain, xgb_params)
    
    # save best model
    n_categories = len(np.unique(df_train['labels']))
    uvpec.save_model(best_model, output_dir, key, n_categories)
    
    # create TAXOCONF file
    MODEL_REF = 'Muvpec_'+key
    uvpec.create_taxoconf(output_dir, dico_id, MODEL_REF, key)

    # evaluate model
    if train_only:
        print('training only, no evaluation')
        sys.exit(0)
    else:
        if(os.path.isfile(os.path.join(output_dir, test_features+'.feather')) == True):
            print('Test features have already been extracted...Loading data')
            dataset_test = pd.read_feather(os.path.join(output_dir, test_features+'.feather'))
            dico_id_test = np.load(os.path.join(output_dir, 'dico_id_test.npy'), allow_pickle=True) # read numpy file
            dico_id_test = dict(enumerate(dico_id_test.flatten(), 1)) # convert numpy ndarray to dict
            dico_id_test = dico_id_test[1] # get the right format for an easy use
        else:
            print("Test features file does not exist...Extracting features...")
            # extraction of features
            dataset_test, dico_id_test = uvpec.extract_features(path_to_test_subfolders, pixel_threshold, objid_threshold_file, use_objid_threshold_file, use_C)
            # save dataset
            dataset_test.to_feather(os.path.join(output_dir, test_features+'.feather'))
            # save dico_id
            np.save(os.path.join(output_dir,'dico_id_test.npy'), dico_id_test)
            print("We are done with the extraction of test features, data have been saved")

        inflexion_filename = os.path.join(output_dir, 'inflexion_point_'+str(key)+'.feather')
        xgb_model = os.path.join(output_dir, 'Muvpec_'+str(key)+'.model')
        uvpec.evaluate_model(n_jobs, dataset_test, xgb_model, inflexion_filename, True, output_dir, use_C)

if __name__ == "__main__":
    main()
