import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# datei has the date if pasager was transported or not, that helps to train the model
tita = pd.read_csv('spaceship_train.csv',index_col=0)
#test of data was readet 
tita.head()

tita.isnull().sum()
print(tita.isnull().sum())

#tita[tita['Name']].isnull()
#print(tita[tita['Name']].isnull())

titanicfill = tita.fillna(value = 0)

print(titanicfill.isnull().sum())


""" imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
SimpleImputer()

X = [[np.nan, 2], [6, np.nan], [7, 6]]

isnull().sum() """