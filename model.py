from traceback import print_tb
import pandas as pd
import numpy as np
import  sklearn 
from sklearn import svm, preprocessing
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score 

# datei has the date if pasager was transported or not, that helps to train the model
df= pd.read_csv('spaceship_train.csv',index_col=0)
# test datei 
testdf= pd.read_csv('spaceship_test.csv',index_col= 0)
#test of data was readet 
df.head()
# all mising values will e replace with a 0


#all not numerical data  wobt work , we need to map then to numbers and the replace it with map 
HomePlanet_map={"Mars":1,"Earth":2,"Europa":3,"0":4}
Destination_map={"TRAPPIST-1e":1 ,"55 Cancri e":2,"PSO J318.5-22":3 }



#titanic["Transported"] = titanic["Transported"].map(transported_map)
df["HomePlanet"] = df["HomePlanet"].map(HomePlanet_map)
df["Destination"] = df["Destination"].map(Destination_map)
#

testdf["HomePlanet"] = testdf["HomePlanet"].map(HomePlanet_map)
testdf["Destination"] = testdf["Destination"].map(HomePlanet_map)

print("test",df.isnull().sum())



test_size= 2000
df = df.fillna(value=df['Age'].mean())
df =  df.fillna(value=df['CryoSleep'].mean())
df = df.fillna(value=df['CryoSleep'].mean())
df= df.fillna(value=df['RoomService'].mean())
df= df.fillna(value=df['FoodCourt'].mean())
df= df.fillna(value=df['ShoppingMall'].mean())
df= df.fillna(value=df['Spa'].mean())
df= df.fillna(value=df['VRDeck'].mean())

df['Cabin'].fillna(method ='pad')
df['HomePlanet'].fillna(method='pad')
df['VIP'].fillna(method="pad")

testdf = testdf.fillna(value=testdf['Age'].mean())
testdf = testdf.fillna(value=testdf['CryoSleep'].mean())
testdf = testdf.fillna(value=testdf['CryoSleep'].mean())
testdf= testdf.fillna(value=testdf['RoomService'].mean())
testdf= testdf.fillna(value=testdf['FoodCourt'].mean())
testdf= testdf.fillna(value=testdf['ShoppingMall'].mean())
testdf= testdf.fillna(value=testdf['Spa'].mean())
testdf= testdf.fillna(value=testdf['VRDeck'].mean())

testdf['Cabin'].fillna(method ='pad')
testdf['HomePlanet'].fillna(method='pad')
testdf['VIP'].fillna(method="pad")


print("fill all holes in testdata")


ti_shuffle= sklearn.utils.shuffle(testdf)
x= df.drop(['Transported','Name','Cabin'], axis=1).values

#what we want to predict 
y=df["Transported"].values

print(testdf.isnull().sum())
test_size= 100

print("fill all holes correct")
print(df.isnull().sum())




#todo null for diferent columns 
# how to use fit 
# print the result from model 
#test tree
#x_train, y_train = titanic(return_x_y = True)
x_train =x[:-test_size]
y_train= y[:-test_size]
x_test =x[-test_size]
y_test =y[-test_size]
clf = HistGradientBoostingClassifier().fit(x_train, y_train)
print("model complete")
x_test = x_test.reshape(1,-1)

clf.score(x_train,y_train)
 
print(clf.score(x_train,y_train))

# cross cvalidator (estimator, x and y ,scoring method )
print("cross validation score")

xT= testdf.drop(['Name','Cabin'], axis=1).values

#what we want to predict 
yT=df["Transported"].values

xT=xT[test_size]
yT=y[test_size]

scores = cross_val_score(clf,x,y,cv=2)
print(scores)


"""#classifer
clf= svm.SVR(kernel="linear")
#train 
clf.fit(x_train,y_train)
#0 ist bad and 1perfekt , 
clf.score(x_test,y_test)
for x,y in zip(x_test,y_test):
    print(f"Model:{clf.predict([x][0])},Actual:(y)") """



