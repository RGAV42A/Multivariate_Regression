
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
from sklearn.model_selection import RandomizedSearchCV
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

#prepare X and y
x_shape = df.shape[1]-1
y_shape = df.shape[1]

X = df.iloc[:,0:x_shape]
y = df.iloc[:,y_shape-1:y_shape]
## list with factors
list_factors = X.columns.values
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



### transform X with MinMaxScaler
# define the scaler
scaler = MinMaxScaler()
# fit on the  dataset
X[list_factors] = MinMaxScaler().fit_transform(X[list_factors])
y = MinMaxScaler().fit_transform(y)

print(X.head())
print(y[:5,:])

# remove the colinearity from X
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

X=X[list_factors].values


###################################################

# Algorithms
models = []
models.append(('KNR', KNeighborsRegressor(n_neighbors=5)))
models.append(('HUBER', HuberRegressor()))
models.append(('LR', LinearRegression()))
models.append(('DTR', tree.DecisionTreeRegressor()))
# define the evaluation method
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=200)

r2_results = []
mse_results = []
names = []
for name,model in models:
    fs = SelectKBest()
    pipeline = Pipeline(steps=[('anova',fs), ('model', model)])
    # define the grid
    grid = {'anova__k':[i+1 for i in range(X.shape[1])],'anova__score_func':[f_regression,mutual_info_regression]}
    # define the grid search
    search = RandomizedSearchCV(pipeline, grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv,n_iter=20,random_state=200)
    # perform the search
    search.fit(X, y.ravel())
    print('\nModel name:',name)
    print("Best: %f using %s" % (search.best_score_, search.best_params_))
    mse_result = cross_val_score(search.best_estimator_, X, y.ravel(), cv=cv, scoring='neg_mean_squared_error')
    r2_result = cross_val_score(search.best_estimator_, X, y.ravel(), cv=cv, scoring='r2')
    # convert scores to positive
    mse_result = np.absolute(mse_result)
    mse_results.append(mse_result)
    r2_results.append(r2_result)
    names.append(name)
    msg = "%s: %f (%f)" % (name, np.mean(mse_results), np.std(mse_results))
    print(msg)
    msg = "%s: %f (%f)" % (name, np.mean(r2_results), np.std(r2_results))
    print(msg)

    # Compare Algorithms
    fig = pyplot.figure()
    fig.suptitle('Model Comparison')

    ax = fig.add_subplot(121)
    ax.set_title('R2')
    pyplot.boxplot(r2_results)
    ax.set_xticklabels(names)
    ax = fig.add_subplot(122)
    ax.set_title('MSE')
    pyplot.boxplot(mse_results)
    ax.set_xticklabels(names)
    pyplot.tight_layout()
    pyplot.savefig('features_model_selection.png',dpi=100)



