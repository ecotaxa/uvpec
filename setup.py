import setuptools

# help here : https://github.com/pypa/sampleproject/blob/main/setup.py and https://packaging.python.org/guides/distributing-packages-using-setuptools/

setuptools.setup(
        # metadata
        name='UVP6foo',
        version='0.0',
        description="Train UVP6 model",
        #long_description=file: README.md, # test if it works that way later
        #long_description_content_type='text/markdown',
        url='',
        author='',
        author_email='',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Programming Language :: Python :: 3'
        ],
        # content
        packages=setuptools.find_packages(),
        entry_points={
            'console_scripts': ['UVP6foo = UVP6foo.__main__:main']
         },
        package_data={
            'UVP6foo': ['config.yaml'],
         },
        python_requires='>=3.6', # check if that is true
        install_requires=[
            'numpy==1.19.5', # check for sup sign
            'pandas==1.2.1',
            'PyYaml==5.3.1',
            'scikit-image==0.18.1',
            'xgboost==1.3.3',
            'sklearn==0.0',
            'pyarrow==3.0.0'
        ]
)

        
