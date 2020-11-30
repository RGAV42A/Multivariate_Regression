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


###################################################
# define number of features to evaluate
num_features = [i for i in range(2, X.shape[1])]
knr = KNeighborsRegressor(n_neighbors=5)
ransac = RANSACRegressor()
huber = HuberRegressor()
lr = LinearRegression()
clf = tree.DecisionTreeRegressor()
models = [lr,knr,clf,huber]

score_funcs = [f_regression,mutual_info_regression]
#score_funcs = [mutual_info_regression]
scfunc_iy = range(len(score_funcs))
best_score = 999
# enumerate each number of features
results = list()
ix = 1
for model in models:
    for score_func in score_funcs:
        print(str(model),str(score_func))
        for k in num_features:
            # create pipeline
            model = model
            fs = SelectKBest(score_func=score_func, k=k)
            #fs.fit(X,y)
            #mask = fs.get_support(indices=True)
            #print('list with features with Select KBest',list(map(lambda x :rfactors[x],mask)))
            pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
            # evaluate the model
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            scores = cross_val_score(pipeline, X, y.values.ravel(), scoring='neg_root_mean_squared_error', cv=cv,n_jobs=-1)
            # convert scores to positive
            scores = np.absolute(scores)
            scor = np.mean(scores)
            if scor < best_score:
                best_score = scor
                score_list = []
                score_list.append('{},{} with {} features and score {} '.format(str(model),str(score_func),k,best_score))
            results.append(scores)
            # summarize the results
            print('>%d %.3f (%.3f)' % (k, np.mean(scores), np.std(scores)))
        # plot model performance for comparison
        pyplot.figure(1,figsize = (10,60))
        pyplot.subplot(9,1,ix)
        pyplot.title('{} {}'.format(str(model),str(score_func)))
        pyplot.boxplot(results, labels=num_features, showmeans=True)
        pyplot.tight_layout()
        pyplot.savefig('SelectKBest.png')
        results = list()
        ix+=1
    print(score_list)

# 'KNeighborsRegressor(),function f_regression with 6 features
