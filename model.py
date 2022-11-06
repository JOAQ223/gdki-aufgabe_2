from traceback import print_tb
import pandas as pd
import numpy as np
import  sklearn 
from sklearn import svm, preprocessing
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV

titanic = pd.read_csv('spaceship_train.csv',index_col=0)
titanic.head()


# ML ist linear Algebra 
#conver everything to numerical
titanic["Transported"].unique
titanic["Transported"].astype("category").cat.codes


print(titanic["Transported"].astype("category").cat.codes)
#all not numerical data map machen and to numbers all diferent values parcing 

transported_map= {"False":1,"True":2}
HomePlanet_map={"Mars":1,"Earth":2,"Europa":3,"":4}
CryoSleep_map={"False":1,"True":2}
Destination_map={"TRAPPIST-1e":1 ,"55 Cancri e":2,"PSO J318.5-22":3 }


titanic["Transported"] = titanic["Transported"].map(transported_map)
titanic["HomePlanet"] = titanic["HomePlanet"].map(HomePlanet_map)
titanic["Destination"] = titanic["Destination"].map(Destination_map)

ti_shuffle= sklearn.utils.shuffle(titanic)

x= titanic.drop(['Transported','Name'], axis=1).values
y=titanic["Transported"].values
#x, y = load:passengenrs(return_x_y = True)
test_size= 6000
x_train =x[:-test_size]
y_train= x[:-test_size]



