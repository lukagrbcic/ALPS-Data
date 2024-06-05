import numpy as np
import sys


sys.path.insert(0, '../benchmarks')

import benchmarks_functions as bf


benchmark = 'logistic_growth'
w = np.linspace(0, 10, 50)

target = bf.targets(benchmark).get_target()

f_ = bf.function(benchmark, target)
lb, ub = f_.get_bounds()

def f(x):
    return f_.evaluate(x)
