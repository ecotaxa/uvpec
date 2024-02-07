import setuptools
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
# help here : https://github.com/pypa/sampleproject/blob/main/setup.py and https://packaging.python.org/guides/distributing-packages-using-setuptools/

setup(
        # metadata
        name='uvpec',
        version='1.0.0',
        description="Train UVP6 model",
        #long_description=file: README.md, # test if it works that way later
        #long_description_content_type='text/markdown',
        url='https://github.com/ecotaxa/uvpec',
        author='Florian Ricour',
        author_email='uvpec.5ppdz@aleeas.com',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Programming Language :: Python :: 3'
        ],
        # content
        packages=setuptools.find_packages(),
        entry_points={
            'console_scripts': ['uvpec = uvpec.__main__:main']
         },
        package_data={
            'uvpec': ['config.yaml'],
         },
        python_requires='>=3.6', # check if that is true
        ext_modules=cythonize([
            Extension("cython_uvp6", ["src/cython_uvp6.pyx"], language="c++")]),
        include_dirs=[numpy.get_include()], # had to add it because of an error on my local machine. See also here: https://stackoverflow.com/questions/14657375/cython-fatal-error-numpy-arrayobject-h-no-such-file-or-directory
        install_requires=[
            'numpy>=1.20.3',
            'pandas>=1.2.1',
            'PyYaml>=5.3.1',
            'scikit-image>=0.18.1',
            'xgboost>=1.3.3',
            'scikit-learn>=1.3.0',
            'pyarrow>=7.0.0',
            'key-generator>=1.0.3',
            'pillow>=8.2.2',
            'matplotlib>=3.3.4',
            'seaborn>=0.11.1'
        ]
)
