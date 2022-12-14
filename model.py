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
testdf= pd.read_csv('spaceship_valid.csv',index_col= 0)
#test of data was readet 
df.head()


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

#fill df with not NAN values in train


test_size= 1000
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

#fill df with not NAN values in valid

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
test_size= 1000

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
#x_test = x_test.reshape(1,-1)
print("model score train")
clf.score(x_train,y_train)
 
print(clf.score(x_train,y_train))

# cross cvalidator (estimator, x and y ,scoring method )
xT= testdf.drop(['Name','Cabin','Transported'], axis=1).values

#what we want to predict 
yT=testdf["Transported"].values
print("model testing with test data");
print(clf.score(xT,yT))

# print(f"Model:{clf.predict([yT])},Actual:(y)")
#clf.predict(yT)
file = open("readme.txt", "w")
with open('readme.txt', 'w') as f:
  for i in range(0,1):
    entry = "Name"+": "+testdf['Name'].index[i]+"transported: "+testdf['Transported'].to_string() +"\n"
    file.write(entry)


file.close()

#xT=xT[-test_size]
#yT=yT[-test_size]

#scores = cross_val_score(clf,x,y,cv=2)





