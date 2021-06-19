# House Price Prediction in California, USA
"""
Created on Thu Apr  1 08:51:31 2021

@author: Aiyub
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV
from datetime import datetime

# Importing the dataset
dataset = pd.read_csv('California housing.csv')
Head = dataset.head()

# Data Visualization on the the whole dataset
dataset.hist(bins=50, figsize=(20,15))
plt.show()

# Information of the whole dataset
info = dataset.info()
missing_values = dataset.isnull().sum()
Shape = dataset.shape
total_null = dataset[dataset['total_bedrooms'].isnull()]

# describing the whole dataset
Describe = dataset.describe()

# Data Preprocessing
Mein = dataset['total_rooms'].mean()
Median = dataset['total_bedrooms'].median()
assign = dataset['total_bedrooms'].fillna(dataset['total_bedrooms'].median(), inplace=True)

# Creat new variables after Data Preprocessing
after_missing_values = dataset.isnull().sum()
after_total_Null = dataset[dataset['total_bedrooms'].isnull()]
after_info = dataset.info()

# Dealing as the ocean proximity
Ocean = dataset["ocean_proximity"].value_counts()
enc = preprocessing.LabelEncoder()
dataset['ocean_proximity']= enc.fit_transform(dataset['ocean_proximity'])
dataset['ocean_proximity'].value_counts()
dataset.head()

# data selection in the dataset
X = dataset.drop('median_house_value', axis=1).values
y = dataset['median_house_value'].values

# Data Scaling
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
X.std()

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

# Training the Multiple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# accuracy the regression
accuracy = regressor.score(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)




# K- Nearest Neighbors Regressor
params = {
    'n_neighbors': [9],  
    'weights': ['distance'],  
    'p': [1]   
    }
knn = KNeighborsRegressor()
rs = GridSearchCV(estimator=knn, param_grid=params, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
rs.fit(X_train, y_train)
print(rs.best_estimator_)
knn = rs.best_estimator_
start = datetime.now()
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
stop = datetime.now()
delta = stop - start
print('-'*30)
r2 = r2_score(y_test, pred)
print('R2: ', r2)
err = np.sqrt(mean_squared_error(y_test, pred))
print('Root Mean Squared Error: ', err)
seconds = delta.seconds + delta.microseconds/1E6
print('Time to compute: ', seconds, 'seconds')

knn_reg = ('KNN', r2, err, seconds)
 


# Support Vector Regressor (SVR)

svr = SVR(C=100, gamma=1, kernel='linear')
start = datetime.now()
svr.fit(X_train, y_train)
pred = svr.predict(X_test)
stop = datetime.now()
delta = stop - start

print('-'*30)
r2 = r2_score(y_test, pred)
print('R2: ', r2)
err = np.sqrt(mean_squared_error(y_test, pred))
print('Root Mean Squared Error: ', err)
seconds = delta.seconds + delta.microseconds/1E6
print('Time to compute: ', seconds, 'seconds')

support_vector_reg = ('SVR', r2, err, seconds)




# Decision Tree Regressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=22) #getting not scaled data

params = {'max_depth': [7], 
          'max_features': ['auto', 'sqrt'], 
          'min_samples_leaf': [7],
          'min_samples_split': [0.1], 
          'criterion': ['mse'] 
         }

tree = DecisionTreeRegressor()
rs = GridSearchCV(estimator=tree, param_grid=params, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
rs.fit(X_train, y_train)
print(rs.best_estimator_)

tree = rs.best_estimator_
start = datetime.now()
tree.fit(X_train, y_train)
pred = tree.predict(X_test)
stop = datetime.now()
delta = stop - start

print('-'*30)
r2 = r2_score(y_test, pred)
print('R2: ', r2)
err = np.sqrt(mean_squared_error(y_test, pred))
print('Root Mean Squared Error: ', err)
seconds = delta.seconds + delta.microseconds/1E6
print('Time to compute: ', seconds, 'seconds')

decision_tree = ('Tree', r2, err, seconds)



     
         





















