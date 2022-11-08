df = pd.read_csv('spaceship_train.csv',index_col=0)
print(df.head(5))

 # Creating the classifier object
def train_using_gini(X_train, X_test, y_train):
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)

   

  # Separating the target variable
    X = df.drop(['Transported','Name','Cabin'], axis=1).values
    Y = df['Transported']

print("fill all holes")
print(df.isnull().sum())
df =  df.fillna(0)
# Splitting the dataset into train and test
#X_train, X_test, y_train, y_test = train_test_split( 
    #X, Y, test_size = 100, random_state = 100)
      #return X, Y, X_train, X_test, y_train, y_test