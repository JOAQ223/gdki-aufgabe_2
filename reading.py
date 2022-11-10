import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline 
a=["a", "b"]
b= [True, False,True, False]
with open('readme.txt', 'w') as f:
    for r in a,b:
        f.write("Name", a)
        f.write("Transported",b)
        f.write('\n')