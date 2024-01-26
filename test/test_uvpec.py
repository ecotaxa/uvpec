import os
import glob

# run package (so it means that the package should be installed before... not sure it is the right way to go ..)
# check if running the config file works (should be improved because there are plenty of steps..)
def test_package():
    os.system('uvpec ./config.yaml')
    assert(os.path.isdir('../test_output')) == True

# check if features file exists
def test_feature():
    assert(os.path.isfile('../test_output/features_train.feather')) == True

# check if inflexion data exists
def test_inflexion():
    arr = glob.glob('../test_output/inflexion_point*.feather')
    assert(len(arr)>0) == True

# check if models are created
def test_models():
    arr = glob.glob('../test_output/Muvpec_*.model')
    assert(len(arr)>0) == True
