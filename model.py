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
titanic= pd.read_csv('spaceship_train.csv',index_col=0)
#test of data was readet 
titanic.head()
# all mising values will e replace with a 0


#all not numerical data  wobt work , we need to map then to numbers and the replace it with map 
transported_map= {"False":1,"True":2}
HomePlanet_map={"Mars":1,"Earth":2,"Europa":3,"0":4}
CryoSleep_map={"False":1,"True":2}
Destination_map={"TRAPPIST-1e":1 ,"55 Cancri e":2,"PSO J318.5-22":3 }


#titanic["Transported"] = titanic["Transported"].map(transported_map)
titanic["HomePlanet"] = titanic["HomePlanet"].map(HomePlanet_map)
titanic["Destination"] = titanic["Destination"].map(Destination_map)

ti_shuffle= sklearn.utils.shuffle(titanic)

print("test",titanic.isnull().sum())

x= titanic.drop(['Transported','Name','Cabin'], axis=1).values

#what we want to predict 
y=titanic["Transported"].values

test_size= 2000

titanic =  titanic.fillna(0)
print("fill all holes")
print(titanic.isnull().sum())
#x_train, y_train = titanic(return_x_y = True)
x_train =x[:-test_size]
y_train= y[:-test_size]
x_test =x[-test_size]
y_test =y[-test_size]
clf = HistGradientBoostingClassifier().fit(x_train, y_train)
print("model complete")
#for x,y in zip(x_test,y_test):
 #   print(f"Model:{clf.predict([x][0])},Actual:(y)") 

""" x_train =x[:-test_size]
y_train= x[:-test_size]


x_test =[-test_size]
y_test = [-test_size]

#classifer
clf= svm.SVR(kernel="linear")
#train 
clf.fit(x_train,y_train)
#0 ist bad and 1perfekt , 
clf.score(x_test,y_test)
for x,y in zip(x_test,y_test):
    print(f"Model:{clf.predict([x][0])},Actual:(y)") """



