# UVP6_classif
Toolbox to train automatic classification models for UVP6 images.

### How to install the package on your machine and test it in 'developer mode' ?
The main interest of the developer mode is to :
1. Avoid to install your unfinished package randomly on your computer, making it easy to just remove it once you are done.
2. It __should__ be easy to uninstall it without any tracks left on your machine.

Now, how do we proceed?

Just git clone the project and write `setyp.py develop` in your terminal. Bingo ! You have now a great UVP6foo package installed on your computer, congratulations !

### How do we use the package?

Just make a nice `config.yaml` file containing the name of the folder where you want to keep the outputs and the name of the folder containing the image subfolders.
Then, write in your terminal `UVP6foo config.yaml` and wait for the magic to happen !

Once it is done, you should have everything you need in the output folder you specified. 

### Last but not least

We have prepared a `test` folder in our package. This allows you to check some stuff at the moment but it should be improved in the near future. To use it,
just navigate in the test folder using `cd test` and then just run `pytest` in your terminal, everything should be now taken care of and if you only see green lights
it means that all tests went smoothly!
