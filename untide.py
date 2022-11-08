from traceback import print_tb
import pandas as pd
import numpy as np
import  sklearn 
from sklearn import svm, preprocessing
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier

# datei has the date if pasager was transported or not, that helps to train the model
df2= pd.read_csv('spaceship_train.csv',index_col=0)
#test of data was readet 
df2.head()
# all mising values will e replace with a 0
#all not numerical data  wobt work , we need to map then to numbers and the replace it with map 
transported_map= {"False":1,"True":2}
HomePlanet_map={'Mars':1,'Earth':2,"Europa":3}
CryoSleep_map={"False":1,"True":2}
Destination_map={"TRAPPIST-1e":1 ,"55 Cancri e":2,"PSO J318.5-22":3 }


#titanic["Transported"] = titanic["Transported"].map(transported_map)
df2["HomePlanet"] = df2["HomePlanet"].map(HomePlanet_map)
df2["Destination"] = df2["Destination"].map(Destination_map)



print("test",df2.isnull().sum())



test_size= 2000
df2 = df2.fillna(value=df2['Age'].mean())
df2 =  df2.fillna(value=df2['CryoSleep'].mean())
df2 = df2.fillna(value=df2['CryoSleep'].mean())
df2= df2.fillna(value=df2['RoomService'].mean())
df2= df2.fillna(value=df2['FoodCourt'].mean())
df2= df2.fillna(value=df2['ShoppingMall'].mean())
df2= df2.fillna(value=df2['Spa'].mean())
df2= df2.fillna(value=df2['VRDeck'].mean())

df2['Cabin'].fillna(method ='pad')
df2['HomePlanet'].fillna(method='pad')
df2['VIP'].fillna(method="pad")


print("fill all holes")


ti_shuffle= sklearn.utils.shuffle(df2)
x= df2.drop(['Transported','Name','Cabin'], axis=1).values

#what we want to predict 
y=df2["Transported"].values

print(df2.isnull().sum())
test_size= 1000

print("model starting")
x_train =x[:-test_size]
y_train= y[:-test_size]


x_test =[-test_size]
y_test = [-test_size]

#classifer
clf= svm.SVR(kernel="linear")
#train 
clf.fit(x_train,y_train)
print("model finishc")
