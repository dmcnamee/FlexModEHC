## Introduction

This code was used for studying the **flexible modulation of sequence generation in the entorhinal-hippocampal system**.

* [Nature Neuroscience](https://www.nature.com/articles/s41593-021-00831-7)
* [Non-paywall version (view only)](https://rdcu.be/ciybE)



## Installation using Anaconda
```
conda create -n FlexModEHC python=3.8
conda activate FlexModEHC
conda install numpy scipy pandas seaborn networkx scikit-learn numba
conda install -c conda-forge gym
conda install -c pyviz holoviews
pip install git+https://github.com/zuoxingdong/mazelab.git
pip install git+https://github.com/dmcnamee/FlexModEHC.git
```

FIGURE_S8 requires torch and opencv
```
conda install pytorch
conda install -c conda-forge opencv
```

## Explanation
* Package is based on a set of core classes which form a chain of inheritances:  
  ENVIRONMENT -> GENERATOR -> PROPAGATOR -> SIMULATOR -> EXPLORER/LEARNER
* A GENERATOR is constructed from an ENVIRONMENT. For example, an environment transition matrix may be used to form a generator matrix.
* A PROPAGATOR takes a GENERATOR (along with several parameters as arguments) and uses eigen-decompositions to form a matrix solution to the master equation defined by the GENERATOR.
* A SIMULATOR uses a PROPAGATOR to sample trajectories in the ENVIRONMENT.
* An EXPLORER samples trajectories from SIMULATOR and performs search process analyses.
* A LEARNER samples trajectories from SIMULATOR and learns internal models and reward functions.
* Each of these classes comes equipped with documented member functions.
