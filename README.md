# AI Driven Laser Parameter Search (ALPS) - Data
Data for the manuscript "AI Driven Laser Parameter Search: Inverse Design of Photonic Surfaces using Greedy Surrogate-based Optimization" by Luka Grbcic (LBNL), Minok Park (LBNL), Juliane Müller (NREL), Vassilia Zorba (LBNL, UCB) and Wibe Albert de Jong (LBNL), 2024.

LBNL - Lawrence Berkeley National Laboratory

NREL - National Renewable Energy Laboratory

UCB - University of California, Berkeley

**Data used to train the experimental model can be found at:**

https://drive.google.com/file/d/19mjGG5-SsU3jjiO2gkQ5M0mhtEGhZ7nu/view?usp=drive_link

ss_data -- Stainless steel data

inconel_data -- Inconel data

Both datasets have been shuffled split into train and test data.

_______

DATA DESCRIPTION -- Models
_________

Pretrained models needed for the Inconel and Stainless steel photonic inverse design benchmarks are given in each respective folder.

**inconel_model.pkl** - Forward model that predicts the Inconel pca components based on the input laser parameters

**inconel_pca.pkl** - PCA model that inversely transforms the pca components to Inconel spectral emissivity curves

**ss_model.pkl** - Forward model that predicts the Stainless steel pca components based on the input laser parameters

**ss_pca.pkl** - PCA model that inversely transforms the pca components to Stainless steel spectral emissivity curves
_______

CODE DESCRIPTION -- Benchmarks
_________

**benchmarks_functions.py** - Python code that contains the benchmark target and functions classes.

Test example **test_function.py** explained below:

```python

import numpy as np
import sys

sys.path.insert(0, '../benchmarks')

#load the benchmark functions module
import benchmarks_functions as bf


benchmark = 'logistic_growth' #select the benchmark by name

target = bf.targets(benchmark).get_target() #get target array of the benchmark if necessary

f_ = bf.function(benchmark, target) #define the function object
lb, ub = f_.get_bounds() #get lower and upper bounds (arrays) of the benchmark

#define the objective function that returns a single value
def f(x): 
    return f_.evaluate(x) #use the evaluate method that takes in the design vector
```
