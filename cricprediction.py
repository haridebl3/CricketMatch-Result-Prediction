import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from sklearn.preprocessing import LabelEncoder
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

# Read data
df = pd.read_csv('Match.csv',nrows=20 )

print(df.head())
print(df.shape)

Y=df['Match_Winner_Id']
print(Y.head())

df.drop(['Match_Winner_Id'],axis=1)
'''
print('After droping Match Winner')
print(df.head(2),df.shape)

'''
#Encode to numerical data
l=LabelEncoder()
df['date']=l.fit_transform(df['Match_Date'])
df['venue']=l.fit_transform(df['Venue_Name'])
df['toss_dec']=l.fit_transform(df['Toss_Decision'])
df['margin']=l.fit_transform(df['Win_Type'])
df['city']=l.fit_transform(df['City_Name'])

print('After Encoding')
print(df.head(2),df.shape)

X=df.drop(['Match_Date','Venue_Name','Toss_Decision','Win_Type','City_Name','Host_Country','Match_Winner_Id'],axis=1)
'''
print('After droping')
print(df.head(2),df.shape)
'''
print(X.head())


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.35,random_state=42)
y_train = y_train.ravel()
y_test = y_test.ravel()

print('Training dataset shape:', X_train.shape, y_train.shape)
print('Testing dataset shape:', X_test.shape, y_test.shape)

# Build RF classifier to use in feature selection
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# Build step forward feature selection
sfs1 = sfs(clf,k_features=6,forward=True,floating=False,verbose=2,scoring='accuracy',cv=2)

# Perform SFFS
sfs1 = sfs1.fit(X_train, y_train)
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)

print(df.columns[feat_cols])

x=df[df.columns[feat_cols]]

print(x.head())
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.35,random_state=42)

print('-----------------Random Forest Classifier------------------------------------')

rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)

rf.fit(X_train,y_train)

predictions = rf.predict(X_test)

errors = abs(predictions - y_test)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors / y_test)


accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')

