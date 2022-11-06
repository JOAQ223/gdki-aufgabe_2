from traceback import print_tb
import pandas as pd
import numpy as np
import  sklearn 
from sklearn import svm, preprocessing

poke = pd.read_csv('pokemon_data.csv')

#print(poke.tail(5))

df_xlxs = pd.read_excel('pokemon_data.xlsx')
#print(df_xlxs.head(3))


df = pd.read_csv('pokemon_data.txt', delimiter='\t')
#print(df.head)

#print(df.columns)

print(df[['Name', 'Type 1', 'HP']][0:5])



a = np.array([1, 2, 3])
df['Total'] = df['HP']+df['Attack']+ df['Defense'] +df['Sp. Def'] +df['Sp. Atk']+df['Speed']
 
print(df.head(5))
#df.drop(columns=['Total'])
df['Total']= df.iloc[:, 4:10].sum(axis=1)
#axis 1 horizontal , 0 vertical


#df = pd.read_csv("datasets/diamonds.csv", index_col=0)
# ML ist linear Algebra 
#conver everything to numerical
#df["cut"].unique()
#df["cut"].astype("category").cat.codes converts everything to numbers

#create  from numbers categories, because its linear algebra , it need numbers 
#for us the same to do with the destination column(atribute)
cut_class_dict= {"fair":1, "Good":2}
#cut to map all the cateogires but now with numbers
df['cut'] = df['cut'].map(cut_class_dict)
# sort , besser ist to schuffle to train 
df= sklearn.utils.shuffle(df)
# list fo features except from preice 
#black list all the columns that u want to block with x drop
x= df.drop("preice", axis=1).values
x = preprocessing.scale(x) # features , list of features  , verything except from lost scale our data 
y = df['price'].values # labels  lost 
test_size= 200
#train data
x_train= x[:-test_size]
y_train= x[:-test_size]
#test data
x_test =[-test_size]
y_test = [-test_size]
#classifer
clf= svm.SVR(kernel="linear")
#train 
clf.fit(x_train,y_train)
#0 ist bad and 1perfekt , 
clf.score(x_test,y_test)
for x,y in zip(x_test,y_test):
    print(f"Model:{clf.predict([x][0])},Actual:(y)")