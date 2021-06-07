# Underwater Vision Profiler Embedded Classifier

Toolbox to train automatic classification models for UVP6 images.

### How to install the package on your machine and test it in 'developer mode' ?

The main interest of the developer mode is to :

1. Avoid to install your unfinished package randomly on your computer, making it easy to just remove it once you are done.
2. It __should__ be easy to uninstall it without any tracks left on your machine.

Now, how do we proceed?

Just git clone the project and write `python setup.py develop` in your terminal (in your remote uvpec folder). Bingo ! You have now a great uvpec package installed on your computer, congratulations !

### How do we use the package?

Just make a nice `config.yaml` file containing the name of the folder where you want to keep the outputs and the name of the folder containing the image subfolders. /!\ The folders containing the images should have names that strictly fit with the EcoTaxa nomenclature (see the .csv for that). /!\
Then, write in your terminal `uvpec config.yaml` and wait for the magic to happen !

Once it is done, you should have everything you need in the output folder you specified. 

### Last but not least

We have prepared a `test` folder in our package. This allows you to check if the pipeline works without lauching a full process that will take a good amount of time. It is always a good idea to check if everything works before using it on a full training set and also after some package updates. To use it,
just navigate in the test folder using `cd test` then run `uvpec config.yaml`. You should see something going on in your terminal.

To check if the pipeline is not broken somewhere, we have implemented some tests that check (so far) if the desired outputs are present at the end of the procedure. If not, that means something went wrong and the error messages can help us find where the leak is. For that,  run `pytest` in your terminal, everything should now be taken care of and if you only see green lights it means that all tests went smoothly!

Just a reminder, if you see some errors during the test, check if you did not forget to run `uvpec config.yaml`.  
`pytest` is not automatically present on your laptop. To install it, `pip install --user pytest`

##### For more information on the developer mode
https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install

##### For a simple installation (not in developer mode) -- OR to update the package (/!\ you should first `pip uninstall uvpec` before reinstalling it)
Run `pip3 install --user git+https://github.com/ecotaxa/uvpec`
