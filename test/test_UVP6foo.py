import os

# run package (so it means that the package should be installed before... not sure it is the right way to go ..)
# check if running the config file works (should be improved because there are plenty of steps..)
def test_package():
	os.system('UVP6foo ./config.yaml')
	assert(os.path.isdir('../test_output')) == True

# check if features file exists
def test_feature():
	assert(os.path.isfile('../test_output/features.feather')) == True

cfg_params = '0.2_5_0.0_False_20'

# check if inflexion data exists
def test_inflexion():
	assert(os.path.isfile('../test_output/inflexion_point_'+cfg_params+'.feather')) == True

# check if models are created
def test_models():
	assert(os.path.isfile('../test_output/xgboost_'+cfg_params+'.model')) == True
	assert(os.path.isfile('../test_output/xgboost_'+cfg_params+'_model.txt')) == True











