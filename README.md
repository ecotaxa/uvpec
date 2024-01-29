# Underwater Vision Profiler Embedded Classifier

Toolbox to train automatic classification models for UVP6 images and/or to evaluate their performances.

Minimal knowledge in python, git and machine learning is needed.

For smooth operation of the toolbox, the toolbox python package must be installed and the toolbox git repository cloned.

This toolbox has been tested on MacOS and Linux (e.g. Ubuntu 20.04/22.04 and Mint 21). We do not garantee it will work on Windows.

### How to install the package ?

First, you need to make sure that you have the two python libraries `setuptools` and `cython (version 0.29.30)` installed on your computer. If you do not have them, run `pip3 install --user setuptools Cython==0.29.30`

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
  - An image directory, where your well organized folders with plankton images are: it is the training set. The plankton images must be sorted by taxonomist classes into subfolders. It is standardized to be used with Ecotaxa. Each subfolder is named by the class's display name, and the ecotaxa ID, separated by two "_", and contains images from only its taxo class : 'DisplayName__EcotaxaID'. The typical way to export data from ecotaxa in such folders organization is to make a D.O.I. export, exporting all images and keep only 'white on black' images = *_100.png. The maximum number of accepted classes is 40.
  - The name of your features file. If it does not already exist, it will be created so give it a great name !
  - The path to a test set for evaluation. Unused if `train_only`is 'true'.
  - The path to a model (extension should be .model, a model created using XGBoost). Only used for `evaluation_only`.
  - The path to a tsv file containing the objid and the UVP6 acquisition threshold of each image for which features will be extracted. Only used if `use_objid_threshold_file` is set to `true`.

For the instrument parameter, you need to specify:
  - The pixel threshold of your UVP6 `uvp_pixel_threshold`, that is the threshold value used to split image pixels into foreground (> threshold) and background (<= threshold) pixels. It is usually comprised between 20 and 22.
  - If you wish to use a variable threshold value (e.g. if you are working with images acquired with different UVP6), set `use_objid_threshold_file` to `true`.

Then, for XGBoost parameters of the training, you need to specify:
  - An initialization seed `random_state`. It is important if you build multiple models with a different XGBoost configuration. The number is not important, you can keep 42 with trust.
  - A number of CPU cores `n_jobs` that will depend on the computational power of your machine or server.
  - The [learning rate](https://en.wikipedia.org/wiki/Learning_rate). It controls the magnitude of adjustements made to the model's parameter during each iteration of training (i.e. in our model, at each boosting round). A high learning rate may cause the optimization to miss the optimal parameter values (e.g. it leads to oscillations or divergence) while a low learning rate might lead to a slow training due to a slow convergence to the minimum of the loss function or it can also get stuck in local minima.
  - The maximum depth of a tree `max_depth`. For technical reasons, it is forbidden to go beyond 7.
  - `weight_sensitivity` represents the weight ($w$) you want to put on biological classes during the training. The minimum value is 0 (i.e. no weight) and the maximum value is 1. It is useful to add a weight to smaller classes because a great number (often $\ge$ 80%) of images from the training set are detritus hence putting $w$ to 0.25 will put more weight on small (biological) classes during training and will force the algorithm to pay more attention to those classes.
  - `detritus_subsampling` can be used if you want to undersample the detritus class in your training. If you think that your detritus class (therefore, you must have one specifically named 'detritus') is too populated (e.g. extreme dataset imbalance) and that removing a part of it is not an issue for your application, then you can fix a given percentage of subsampling for that class. For example, a `subsampling_percentage` of 20 means that you only keep 20% of your entire detritus class. Keep `detritus_subsampling` to 'false' if you don't want to use it.
  - `subsampling_percentage` is the percentage of images of 'detritus' from your training set you want to keep for training. 
  - `num_trees_CV` stands for the number of boosting rounds you want to use for the cross-validation (CV). This is equivalent to the parameter `num_round` in [XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).

You will also notice that there is one last thing. `use_C` gives the possibility to extract the features from images using a C++ version. We advise to keep it to 'true' because it is much faster than the python version.

Once you are done, run `uvpec config.yaml` in your terminal and wait for the magic to happen ! You should get everything you need in the output folder you specified. 

### Last but not least

We have prepared a `test` folder in our package. This allows you to check if the pipeline works without launching a full process that will take a significant amount of time. It is always a good idea to check if everything works well before using it on a full training set and also after some package updates. To use it,
navigate in the test folder using `cd test` then run `uvpec config.yaml`. You should see something going on in your terminal. Don't forget to check your output folder now !

In addition, there is also another test that you can run in order to see if the pipeline is not broken somewhere. For that,  run `pytest` (that actually looks for test_uvpec.py) in your terminal. Everything should now be taken care of and if you only see green lights it means that all tests went smoothly! If not, that means something went wrong and the error messages can help you find where the leak is. 

Just a reminder, if you see some errors during the test, check if you did not forget to run `uvpec config.yaml`.  
`pytest` is not automatically present on your laptop. To install it, `pip install --user pytest`

### How to prepare your dataset from an Ecotaxa project ? 

You can refer to the documentation on [Ecotaxa](https://ecotaxa.obs-vlfr.fr/) to download all the vignettes you need to use for your train and/or test set. See the "export project" part of your project on https://ecotaxa.obs-vlfr.fr/.

Ecotaxa is built with a rest [API](https://ecotaxa.obs-vlfr.fr/api/docs) that has been designed to facilitate the work of the users. Two packages have been developped to interact more easily with the API in [python](https://github.com/ecotaxa/ecotaxa_py_client) and in [R](https://github.com/ecotaxa/ecotaxarapi). 
Be careful to download the vignettes with the black background because every object is stored in two versions: one with a white backgroud and one with a black background. You will also need to remove the size legend at the bottom of the vignette. To do so, crop 31 pixel at the bottom of the vignette.

Finally, just rename the vignettes with the UVPEC standard (i.e. taxa__ID), and you are good to go ! 

### How to uninstall the package ?

Run `pip uninstall uvpec`

### How to update the package ?
Run `pip uninstall uvpec` before reinstalling it then run`pip3 install --user git+https://github.com/ecotaxa/uvpec`
