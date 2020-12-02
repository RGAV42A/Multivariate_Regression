
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
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor

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
y[['MEDV']] = scaler.fit_transform(y[['MEDV']])

print(X.head())
print(y.head())

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

fs = SelectKBest(score_func=f_regression, k=6)
X_new = fs.fit_transform(X,y.values.ravel())

# ensembles
ensembles = []
ensembles.append(('AB',AdaBoostRegressor()))
ensembles.append(('GBM',GradientBoostingRegressor()))
ensembles.append(('RF',RandomForestRegressor(n_estimators=10)))
ensembles.append(('ET',ExtraTreesRegressor(n_estimators=10)))

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
results = []
names = []
for name, model in ensembles:
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=500)
    cv_results = cross_val_score(model, X_new, y, cv=cv, scoring='r2')
    # convert scores to positive
    cv_results = np.absolute(cv_results)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.savefig('ensemble_methods.png',dpi=80)

# AB: 0.806038 (0.085640)
# GBM: 0.847989 (0.093469)
# RF: 0.820097 (0.107935)
# ET: 0.854929 (0.050555)

