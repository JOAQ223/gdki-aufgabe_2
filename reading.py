import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline 

X_new = QuantileTransformer(n_quantiles=100).fit_transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], c=y);


df = pd.read_csv("drawndata1.csv")

df.head(3)

X = df[['x', 'y']].values
y = df['z'] == "a"




plt.scatter(X[:, 0], X[:, 1], c=y);

