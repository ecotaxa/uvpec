import os
import logging
import pkg_resources
import sys
import yaml
import numpy as np

#from ipdb import set_trace as db

def read_config(user_config_file):
    """
    Configure uvpec options
    Args:
        user_config_file (str): path to the user-defined configuration file
    Returns:
        dict: settings in key-value pairs
    """
    # get general logger
    log = logging.getLogger()
    log.debug("read uvpec default configuration")

    # read default config.yaml (provided in the package)
    defaults_file = pkg_resources.resource_filename("uvpec", "config.yaml")
    with open(defaults_file, 'r') as ymlfile:
        defaults_cfg = yaml.safe_load(ymlfile)

    # read user-defined configuration file
    log.debug("read user-defined configuration file")
    with open(user_config_file, 'r') as ymlfile:
        user_cfg = yaml.safe_load(ymlfile)

    log.debug("combine defaults and user-level settings")
    # settings in the user's config will update those in the defaults
    # settings missing in the user's config will be kept at their default values (and added to the user's config after writing the file back)
    cfg = left_join_dict(defaults_cfg, user_cfg)

    # check correctedness of configuration values
    log.debug("check configuration values")
    
    # features_ID
    assert isinstance(cfg['io']['training_features_file'], (str)), 'training_features_file > should be a string'

    # random_state
    assert isinstance(cfg['xgboost']['random_state'], (int)), 'xgboost > random_state should be an integer'	

    # n_jobs
    assert isinstance(cfg['xgboost']['n_jobs'], (int)), 'xgboost > n_jobs should be an integer'
    
    # learning_rate
    assert isinstance(cfg['xgboost']['learning_rate'], (float)), 'xgboost > learning_rate should be a positive float'
    
    # max_depth
    assert isinstance(cfg['xgboost']['max_depth'], (int)), 'xgboost > max_depth shoud be an integer <= 7'
    
    # detritus_subsampling
    assert isinstance(cfg['xgboost']['detritus_subsampling'], (bool)), 'xgboost > detritus_subsampling should be a boolean'
    
    # detritus percentage for subsampling
    assert isinstance(cfg['xgboost']['subsampling_percentage'], (int)), 'xgboost > subsampling_percentage should be in ]0,100['
    
    # weight_sensitivity
    assert isinstance(cfg['xgboost']['weight_sensitivity'], (float)), 'xgboost > weigth_sensitivity should be a positive float'
    
    # num_trees_CV
    assert isinstance(cfg['xgboost']['num_trees_CV'], (int)), 'xgboost > num_trees_cv should be an integer'

    # use C for extracting features
    assert isinstance(cfg['language']['use_C'], (bool)), 'use_C should be a boolean'

    # add the configuration to the log
    log.info(cfg)

    log.debug("write updated configuration file")
    # change yaml dictionnary writer to preserve the order of the input dictionnary instead of sorting it alphabetically
    # https://stackoverflow.com/questions/16782112/can-pyyaml-dump-dict-items-in-non-alphabetical-order
    yaml.add_representer(dict, lambda self, data: yaml.representer.SafeRepresenter.represent_dict(self, data.items()))

    with open(user_config_file, 'w') as ymlfile:
        yaml.dump(cfg, ymlfile, default_flow_style=False)

    return cfg

def left_join_dict(x, y):
    """
    Recursive left join of dictionnaries.
    
    Any element of y whose key is in x will update the value from x.
    Any element of y that does not exists in x is ommited.
    Inspired from https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
    Args:
        x (dict): reference dictionnary
        y (dict): dictionnary to be merged in x
    
    Returns:
        (dict) a new dictionnary.
    """
    # make a copy of the reference dict
    x = x.copy()
    # remove elements from y that are not in x
    y = { k: y[k] for k in set(x).intersection(set(y)) }
    # for each element of y
    for k,v in y.items():
        # if dict, iterate
        if isinstance(x.get(k), dict) and isinstance(v, dict):
            x[k] = left_join_dict(x[k], v)
        # if not, update
        else:
            x[k] = v
    return(x)
