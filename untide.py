import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

numeric_cols= columns_to_use = [

]


data = pd.read_csv('hosueprice.csv',isecols = numeric_cols)

data.tail();

imputer = SimpleImputer