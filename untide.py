from traceback import print_tb
import pandas as pd
import numpy as np
import  sklearn 
from sklearn import svm, preprocessing
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# datei has the date if pasager was transported or not, that helps to train the model
df2= pd.read_csv('spaceship_valid.csv',index_col=0)

#data from validation 
testdf= pd.read_csv('spaceship_valid.csv',index_col= 0)

df2.head()
print(df2.head(5))

#all not numerical data  wobt work , we need to map then to numbers and the replace it with map 
HomePlanet_map={"Mars":1,"Earth":2,"Europa":3,"0":4}
Destination_map={"TRAPPIST-1e":1 ,"55 Cancri e":2,"PSO J318.5-22":3 }


df2["HomePlanet"] = df2["HomePlanet"].map(HomePlanet_map)
df2["Destination"] = df2["Destination"].map(Destination_map)

test_size= 1000
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


ti_shuffle= sklearn.utils.shuffle(testdf)
X= df2.drop(['Transported','Name','Cabin'], axis=1).values

#what we want to predict 
y=df2["Transported"].values
test_size=6000
X_train = X[:-test_size]
y_train = y[:-test_size]

X_test = X[-test_size:]
y_test = y[-test_size:]

clf = SGDRegressor(max_iter=1000)
clf.fit(X_train, y_train)


#X, y = df2(n_samples=100, centers=2, n_features=2, random_state=1)

train_schuffle= sklearn.utils.shuffle(df2)

# all mising values will e replace with a 0
#all not numerical data  wobt work , we need to map then to numbers and the replace it with map 
model = LogisticRegression()
model.fit(X, y)

# new instances where we do not know the answer
testdf= pd.read_csv('spaceship_valid.csv',index_col= 0)


testdf["HomePlanet"] = testdf["HomePlanet"].map(HomePlanet_map)
testdf["Destination"] = testdf["Destination"].map(HomePlanet_map)

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


ti_shuffle= sklearn.utils.shuffle(testdf)

xT= testdf.drop(['Transported','Name','Cabin'], axis=1).values

#what we want to predict 
yT=testdf["Transported"].values

xT,_  = testdf(n_samples=3, centers=2, n_features=2, random_state=1)
#make a prediction
ynew = model.predict(xT)

# show the inputs and predicted outputs
for i in range(0,1):
	print("X=%s, Predicted=%s" % (xT[i], ynew[i]))

file = open("readme.txt", "w")
for i in range(len(xT)):
    entry = "Name"+": "+df2['Name'].index[i]+"transported: "+df2['Transported'].to_string() +"\n"
    file.write(entry)

file.close()
