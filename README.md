# Underwater Vision Profiler Embedded Classifier

Toolbox to train automatic classification models for UVP6 images and/or to evaluate the performances.

Minimal knowledge in python, git and machine learning is needed.

For smooth operation of the toolbox, the toolbox python package must be installed and the toolbox git repository cloned.

### How to install the package ?

First, you need to make sure that you have the two python libraries `setuptools` and `cython` installed on your computer. If you do not have them, run `pip3 install --user setuptools cython`

Then, to install the package, run `python -m pip install git+https://github.com/ecotaxa/uvpec` in a terminal. You can also use the SSH version with `python -m pip install git+ssh://git@github.com/ecotaxa/uvpec.git`.
Bingo ! You have now a great `uvpec` package installed on your computer, congratulations ! You can check if it is installed by running in your terminal `pip list | grep uvpec`

### How to clone the repository ?

Run `git clone https://github.com/ecotaxa/uvpec.git` for HTTPS or 
`git clone git@github.com:ecotaxa/uvpec.git` for SSH.

### How to use the package?

In order to use the `uvpec` package, you have to create a `config.yaml` file. Don't panic, you have an example of such a file in your cloned repository in `uvpec/uvpec/config.yaml`. In the latter, you need to specify 3 things : what do you want to do with the package, some input/output information and the parameters for the gradient boosted trees algorithm (XGBoost) that will train and create a classification model.

For the process information, you need to specify two boolean variables:
  - `evaluate_only` : 'true' if you only want to evaluate an already created model. In that case, the package will not train any model and will do only the evaluation of the model indicated by `model` path with the `test_set` data. 'false' if you want to train a model.
  - `train_only`: 'true' if you want to only train a model and skip the evaluation part. 'false' if not. NOT TAKEN INTO ACCOUNT IF `evaluate_only` is true.

For the input/ouput (io), you need to specify:
  - An output directory, where the model and related information will be exported
  - An image directory, where your well organized folders with plankton images are: it is the training set. The plankton images must be sorted by taxonomist classes into subfolders. It is standardized to be used with Ecotaxa. Each subfolder is named by the class's display name, and the ecotaxa ID, separated by two "_", and contains images from only its taxo class : 'DisplayName__EcotaxaID'. The typical way to export data from ecotaxa in such folders organization is to make a D.O.I. export, exporting all images and keep only 'white on black' images = *_100.png.
  - The name of your features file. If it does not already exist, it will be created so give it a great name !
  - The path to a test set for evaluation. Unused if `train_only`is 'true'.
  - The path to a model. Only used for `evaluation_only`.

Then, for XGBoost parameters of the training, you need to specify:
  - An initialization seed `random_state`. It is important if you build multiple models with a different XGBoost configuration. The number is not important, you can keep 42 with trust.

  - A number of CPU cores `n_jobs` that will depend on the computational power of your machine or server
  - The learning rate
  - The maximum depth of a tree `max_depth`. For technical reasons, it is forbidden to go beyond 7
  - A weight or `weight_sensitivity` that represents the weight you want to put on biological classes during the training because eh, we all know that 90% of images is marine snow
  - `detritus_subsampling` can be used if you want to undersample your training set. Keep it to 'false' if you don't want to use it
  - `subsampling_percentage` is about how much you want to undersample the 'detritus' class of your training set
  - `num_trees_CV` stands for the number of rounds you want to use for the cross-validation. The latter is use to determine the optimal number of rounds before overfitting. The bigger the number, the longer it takes to process

You will also notice that there is one last thing. `use_C` gives the possibility to extract the features from images using a C++ version. We advise to keep it to 'true' because it is much faster than the python version.

Once you are done, run `uvpec config.yaml` in your terminal and wait for the magic to happen ! You should get everything you need in the output folder you specified. 

### Last but not least

We have prepared a `test` folder in our package. This allows you to check if the pipeline works without launching a full process that will take a significant amount of time. It is always a good idea to check if everything works well before using it on a full training set and also after some package updates. To use it,
navigate in the test folder using `cd test` then run `uvpec config.yaml`. You should see something going on in your terminal. Don't forget to check your output folder now !

In addition, there is also another test that you can run in order to see if the pipeline is not broken somewhere. For that,  run `pytest` (that actually looks for test_uvpec.py) in your terminal. Everything should now be taken care of and if you only see green lights it means that all tests went smoothly! If not, that means something went wrong and the error messages can help you find where the leak is. 

Just a reminder, if you see some errors during the test, check if you did not forget to run `uvpec config.yaml`.  
`pytest` is not automatically present on your laptop. To install it, `pip install --user pytest`

### How to uninstall the package ?

Run `pip uninstall uvpec`

### How to update the package ?
Run `pip uninstall uvpec` before reinstalling it then run`pip3 install --user git+https://github.com/ecotaxa/uvpec`
