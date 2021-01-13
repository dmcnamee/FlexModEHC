#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np; np.seterr(all='raise')
import seaborn as sb
import matplotlib.pyplot as plt

verbose         = False # True to output detailed code progress statements
timing          = True
jit_nopython    = True
jit_cache       = True
sparsify        = False
debug           = False

# settings
# min_val = np.finfo(float).eps # sets lower bounds on probabilities
min_val = 10**-5 # sets lower bounds on probabilities
n_decimals = 6 # calculations rounded to n_decimals places to suppress numerical fluctuations

os.environ['NUMBA_DEBUG_ARRAY_OPT_STATS'] = str(1)
os.environ['NUMBA_DISABLE_JIT'] = str(1) # set to 1 to disable numba.jit, otherwise 0
os.environ['NUMBA_WARNINGS'] = str(1)

if not sys.warnoptions and verbose:
    import warnings
    warnings.simplefilter("default") # "default"/"error"
    os.environ["PYTHONWARNINGS"] = "default" # Also affect subprocesses
