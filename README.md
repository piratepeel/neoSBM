# neoSBM
A new type of SBM

For details on usage run:
```
python runNeoSBM.py -h 
```
more details to follow.

# Installing a Python 2.7 environment using conda

neoSBM is written using Python 2.7.  If you attempt to run the usage command above using Python 3, you will receive the following error message

```
python runNeoSBM.py -h 


  File "runNeoSBM.py", line 49
    print '\n\n'
               ^
SyntaxError: Missing parentheses in call to 'print'. Did you mean print('\n\n')?
```

One way to install a Python 2.7 environment without modifying your system Python is to install the free [miniconda installer](https://docs.conda.io/en/latest/miniconda.html).

Once miniconda has been installed, create an environment for neoSBM using the following command in bash (Linux/Mac) or cmd.exe (Windows)

```
conda create -n neoSBM python=2.7 numpy scipy
```

Switch to this environment using the command

```
conda activate neoSBM
```

It should now be possible to run neoSBM from this command prompt.  You can go back to the system version of Python when you are finished with

```
conda deactivate
```
