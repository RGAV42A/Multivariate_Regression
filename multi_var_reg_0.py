#!/usr/bin/env python3.6

## sample analysis with BostonHausing data set

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance

##### READ DATA

df=pd.read_csv('BostonHausing.csv')
print(df.head())

# calculate duplicates
dups = df.duplicated()
# report if there are any duplicates
print('\nany duplicates:',dups.any())
# list all duplicate rows
print('\nlist all duplicate rows:',df[dups])

# summarize the number of unique values in each column
list_test = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
print('\nnumber of unique values in each column:',df[list_test].nunique())

# summarize the number of unique values in each column
for ix in list_test:
    num = len(np.unique(df[ix]))
    percentage = float(num) / df.shape[0] * 100
    print('{}, {}, {}%'.format(ix, num, percentage))

# remove cols with low uniqie numbers
list_test.remove('ZN')


# the variance threshold for feature selection
print('variance threshold for feature selection before:', df[list_test].shape)
# define the transform
transform = VarianceThreshold()
# transform the input data
X_sel = transform.fit_transform(df[list_test])
print('variance threshold for feature selection after',X_sel.shape)

X=df[list_test]
y=df['MEDV']

# summarize the shape of the training dataset
print('\ndataset before outlier cleaning:',X.shape, y.shape)
# identify outliers
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X)
# select all rows that are not outliers
mask = yhat != -1
X, y = X.iloc[mask, :], y.iloc[mask]
# summarize the shape of the updated training dataset
print('dataset after outlier cleaning:',X.shape, y.shape)

### multi var model with MinMaxScaler
# define the scaler
scaler = MinMaxScaler()
# fit on the training dataset
scaler.fit(X)
# scale the training dataset
X = scaler.transform(X)
# scale the test dataset
#X_test = scaler.transform(X_test)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# fit the model
model = KNeighborsRegressor()
# fit the model
model.fit(X_train, y_train)
# perform permutation importance
results = permutation_importance(model, X, y, scoring='neg_mean_squared_error')
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))






