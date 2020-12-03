
import subprocess

#out = subprocess.run(['/bin/bash', '-c','dir'],shell=True)
#print(out)


## sample analysis with BostonHausing data set

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import f_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from scipy.stats import randint
from sklearn.pipeline import Pipeline
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
X[list_factors] = scaler.fit_transform(X[list_factors])
y = scaler.fit_transform(y)

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

X=X[list_factors]

# ensembles
ensembles = []
ensembles.append(('AB',AdaBoostRegressor()))
ensembles.append(('GBM',GradientBoostingRegressor()))
ensembles.append(('RF',RandomForestRegressor()))
ensembles.append(('ET',ExtraTreesRegressor()))

r2_results = []
mse_results = []
names = []

# evaluate model
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=200)

for name, model in ensembles:
    fs = SelectKBest(score_func=f_regression)
    pipeline = Pipeline(steps=[('anova',fs), ('model', model)])
    # define the grid
    grid = {'anova__k':[i+1 for i in range(X.shape[1])],'model__n_estimators':randint(10,400)}
    # define the grid search
    search = RandomizedSearchCV(pipeline, grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv, random_state=500)
    search.fit(X, y.ravel())
    print('\n',name)
    print ('Best Parameters: ', search.best_params_)

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=500)
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
fig.suptitle('Ensemble Algorithm Comparison')
ax = fig.add_subplot(121)
ax.set_title('R2')
pyplot.boxplot(r2_results)
ax.set_xticklabels(names)
ax = fig.add_subplot(122)
ax.set_title('MSE')
pyplot.boxplot(mse_results)
ax.set_xticklabels(names)
pyplot.tight_layout()
pyplot.savefig('ensemble_methods.png',dpi=80)

'''
AB
Best Parameters:  {'anova__k': 6, 'model__n_estimators': 247}
AB: 0.007990 (0.001661)
AB: 0.808937 (0.046910)
 GBM
Best Parameters:  {'anova__k': 7, 'model__n_estimators': 114}
GBM: 0.007206 (0.001943)
GBM: 0.829694 (0.051646)
 RF
Best Parameters:  {'anova__k': 6, 'model__n_estimators': 297}
RF: 0.007102 (0.002031)
RF: 0.832396 (0.051862)
 ET
Best Parameters:  {'anova__k': 6, 'model__n_estimators': 247}
ET: 0.006700 (0.001970)
ET: 0.841891 (0.049272)
'''
