import subprocess

#out = subprocess.run(['/bin/bash', '-c','dir'],shell=True)
#print(out)


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
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import HuberRegressor,RANSACRegressor
from sklearn.feature_selection import f_regression
from sklearn import tree
from sklearn.model_selection import RepeatedKFold
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor
from scipy.stats import randint as sp_randint
import warnings

warnings.simplefilter('ignore')


##### READ DATA
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

#print('df.shape',df.shape)
x_shape = df.shape[1]-1
y_shape = df.shape[1]
#print('x_shape',x_shape)
#print('y_shape',y_shape)
X = df.iloc[:,0:x_shape]
#print(X.head())
y = df.iloc[:,y_shape-1:y_shape]
#print(y.head())
## list with factors
list_factors = X.columns.values
#print(list_factors)
# calculate duplicates
dups = df.duplicated()
# report if there are any duplicates
print('\nany duplicates:',dups.any())
# list all duplicate rows
print('\nlist all duplicate rows:',df[dups])

# summarize the number of unique values in each column

print('\nnumber of unique values in each column:',X.nunique())

# summarize the number of unique values in each column
for ix in list_factors:
    num = len(np.unique(df[ix]))
    percentage = float(num) / df.shape[0] * 100
    print('{}, {}, {}%'.format(ix, num, percentage))

# remove cols with low uniqie numbers

del X['ZN']

## list with factors
list_factors = X.columns.values.tolist()
print('type list factors:',type(list_factors))

# the variance threshold for feature selection
print('variance threshold for feature selection before:', X.shape)
# define the transform
transform = VarianceThreshold()
# transform the input data
X_sel = transform.fit_transform(X)
print('variance threshold for feature selection after',X_sel.shape)


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
# fit on the  dataset
X[list_factors] = scaler.fit_transform(X[list_factors])
y[['MEDV']] = scaler.fit_transform(y[['MEDV']])

print(X.head())
print(y.head())

for i in np.arange(0,len(list_factors)):
    vif = [variance_inflation_factor(X[list_factors].values, ix) for ix in range(X[list_factors].shape[1])]
    maxloc = vif.index(max(vif))
    print('maxloc',maxloc)
    if max(vif) > 10:
        #print('vif :', vif)
        print('dropping ' + X[list_factors].columns[maxloc] + ' at index:  ' + str(maxloc))
        del list_factors[maxloc]
    else:
        break
print('Final variables:', list_factors)


X=X[list_factors]


#create pipeline
fs = SelectKBest(score_func=f_regression, k=6)
X_new = fs.fit_transform(X,y.values.ravel())
mask = fs.get_support(indices=True)
mask = np.asarray(mask)
rfactors = np.asarray(list_factors)
rfactors = rfactors[mask]
print('list with features with Select KBest',rfactors)
# evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=200)

# define the grid
param_dict = {'model__n_neighbors':[2,3,4,5,6,7,8,9]}
model = KNeighborsRegressor()
pipeline = Pipeline(steps=[('SelectKBest',fs),('model',model)])
# run randomized search
n_iter_search = 20
#random_search = RandomizedSearchCV(pipeline, param_distributions=param_dict,cv=cv, n_iter=n_iter_search,verbose=0, n_jobs=-1, random_state=200,scoring='r2')
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dict,cv=cv, n_iter=n_iter_search,verbose=0, n_jobs=-1, random_state=200,scoring='neg_mean_squared_error')
random_search.fit(X_new, y.values.ravel())
print ('Best Parameters: ', random_search.best_params_)
results = cross_val_score(random_search.best_estimator_,X_new,y.values.ravel(),scoring='r2', cv=cv, n_jobs=-1)
print ("R2 KNN: ", results.mean(),results.std())
