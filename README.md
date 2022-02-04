# Underwater Vision Profiler Embedded Classifier

Toolbox to train automatic classification models for UVP6 images.

### How to install the package ?

First, you need to make sure that you have the two python libraries `setuptools` and `cython` installed on your computer. If you do not have them, run `pip3 install --user setuptools cython`

Then, to install the package, run `python -m pip install git+https://github.com/ecotaxa/uvpec` in a terminal. You can also use the SSH version with `python -m pip install git+ssh://git@github.com/ecotaxa/uvpec.git`.
Bingo ! You have now a great `uvpec` package installed on your computer, congratulations ! You can check if it is installed by running in your terminal `pip list | grep uvpec`

### How to clone the repository ?

Run `git clone https://github.com/ecotaxa/uvpec.git` for HTTPS or 
`git clone git@github.com:ecotaxa/uvpec.git` for SSH.

### How to use the package?

In order to use `uvpec` and train classification models for plankton (UVP6) images, you have to create a `config.yaml` file. Don't panic, you have an example of such a file in your cloned repository in `uvpec/uvpec/config.yaml`. In the latter, you need to specify 2 things : some input/output information and the parameters for the gradient boosted trees algorithm (XGBoost) that will train and create a classification model.
For the input/ouput (io), you need to specify:
  - An output directory, where the model and related information will be exported
  - An image directory, where your well organized folders with plankton images are. Ideally, the folders' name (your organisms) should already exist in EcoTaxa. There is a csv in the cloned repository that you can check
  - The name of your features file. If it does not already exist, it will be created so give it a great name !

Then, for the XGBoost parameters, you need to specify:
  - An initialization seed `random_state`. It is important if you build multiple models with a different XGBoost configuration. The number is not important, you can keep 42 with trust.

  - A number of cores `n_jobs` that will depend on the computational power of your machine or server
  - The learning rate
  - The maximum depth of a tree `max_depth`. For technical reasons, it is forbidden to go beyond 7
  - A weight or `weight_sensitivity` that represents the weight we want to put on biological classes during the training because eh, we all know that 90% of images is marine snow
  - `detritus_subsampling` can be used if you want to undersample your training set. Keep it to 'false' if you don't want to use it
  - `subsampling_percentage` is about how much you want to undersample the 'detritus' class of your training set
  - `num_trees_CV` stands for the number of rounds you want to use for the cross-validation. The latter is use to determine the optimal number of rounds before overfitting. The bigger the number, the longer it takes to process

You will also notice that there is one last thing. `use_C` gives the possibility to extract the features from images using a C++ version. We advise to keep it to 'true' because it is much faster than the python version.

Once you are done, run `uvpec config.yaml` in your terminal and wait for the magic to happen ! You should get everything you need in the output folder you specified. 

### Last but not least

We have prepared a `test` folder in our package. This allows you to check if the pipeline works without lauching a full process that will take a good amount of time. It is always a good idea to check if everything works before using it on a full training set and also after some package updates. To use it,
navigate in the test folder using `cd test` then run `uvpec config.yaml`. You should see something going on in your terminal. Don't forget to check your output folder now !

To check if the pipeline is not broken somewhere, we have implemented some tests that check (so far) if the desired outputs are present at the end of the procedure. If not, that means something went wrong and the error messages can help us find where the leak is. For that,  run `pytest` (that actually looks for test_uvpec.py) in your terminal, everything should now be taken care of and if you only see green lights it means that all tests went smoothly!

Just a reminder, if you see some errors during the test, check if you did not forget to run `uvpec config.yaml`.  
`pytest` is not automatically present on your laptop. To install it, `pip install --user pytest`

### How to uninstall or if you want to update the package ?

Run `pip uninstall uvpec` before reinstalling it
Run `pip3 install --user git+https://github.com/ecotaxa/uvpec`
