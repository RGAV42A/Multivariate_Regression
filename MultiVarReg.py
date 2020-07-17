import pandas as pd
import numpy as np
#from pylab import *
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model,feature_selection
import statsmodels.graphics.api as smg
from statsmodels.stats.outliers_influence import variance_inflation_factor,OLSInfluence
import statsmodels.api as sm
import statsmodels.stats.diagnostic as ssd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.feature_selection import RFE

###########################################################

### Multivariate Regression  #####
'''
### data preprocessing
i=0
linelist=[]
titlelist=[]
path='BostonHousingDataset.txt'
hand=open(path)
### read name columns
for line in hand:
    line=line.rstrip()
    if i>6:
        linesp=line.split()
        #print(linesp[0])
        titlelist.append(linesp[0])
    if i>19:  break
    i+=1
#print('titlelist',len(titlelist))

###  read and merge data
j=0
shortline=[]
longline=[]
for line in hand:
    line=line.rstrip()
    linesp=line.split()
    #print(linesp)
    if len(linesp)>0:
        shortline.append(linesp)


    #print(linesp)
    #if j>106: break
    #j+=1
for k in range(0,len(shortline)-1,2):
    ges=shortline[k]+shortline[k+1]
    longline.append(ges)
    #print(len(ges))
#print(longline)

df=pd.DataFrame(longline,columns=titlelist)

df.to_csv('BostonHausing.csv',index=False)
'''
##### READ DATA
dataframe=pd.read_csv('BostonHausing.csv')
'''
#### NORMALITY TEST
nameindex = list(dataframe.columns)
for ix in nameindex:
    p_val=ssd.kstest_normal(dataframe[ix].values)
    print('p-value for {} = {:.4f}'.format(ix,p_val[1]))

##### CREATE CORRELATION MATRIX
corr=dataframe.corr()
###### PLOT CORELATION MATRIX
smg.plot_corr(corr,xnames=list(corr.columns))
plt.savefig('corr_matrix.png',dpi=125)

### MAKE BOXPLOT
listnames = ['CRIM', 'ZN', 'AGE', 'DIS', 'RAD', 'LSTAT']
n=321
for ix in range(len(listnames)):
    plt.subplot(n+ix)
    plt.title('{}'.format(listnames[ix]))
    print('boxplot',listnames[ix])
    plt.boxplot(dataframe[listnames[ix]].values)
plt.tight_layout()
plt.savefig('boxplot.png',dpi=125)

## FEATURE SELECTION WITH VIF
###### REMOVE MULTICOLLINEARITY
listnames = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
#listnames = ['CRIM', 'ZN', 'AGE', 'DIS', 'RAD', 'LSTAT']
## calc VIF

# use the list to select a subset from original DataFrame
X = dataframe[listnames]
Y = dataframe['MEDV']

for i in np.arange(0,len(listnames)):
    vif = [variance_inflation_factor(X[listnames].values, ix) for ix in range(X[listnames].shape[1])]
    maxloc = vif.index(max(vif))
    if max(vif) > 10:
        #print('vif :', vif)
        #print('dropping' + X[listnames].columns[maxloc] + 'at index: ' + str(maxloc))
        del listnames[maxloc]
    else:
        break
print('Final variables:', listnames)

## FEATURE SELECTION WITH Backward Elimination ** PROBLEM: NO REMOVAL OF MULTICOLLINEARITY
listnames = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = dataframe[listnames]
Y = dataframe['MEDV']

cols = listnames
pmax = 1
while (len(cols)>0):
    p= []
    X = dataframe[cols]
    lm = sm.OLS(Y,X).fit()
    p = pd.Series(lm.pvalues.values,index = cols)

    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
print(lm.summary())

#######   Build the Multivariate Linear Regression Model
### LIST WITH FEATURES
#listnames = ['CRIM', 'ZN', 'AGE', 'DIS', 'RAD', 'LSTAT']
listnames = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

### ## FEATURE SELECTION WITH RFE ** PROBLEM: NO REMOVAL OF MULTICOLLINEARITY
X = dataframe[listnames]
Y = dataframe['MEDV']

r_sq=0
#Variable to store the optimum features
nof=0
#no of features
nof_list=np.arange(1,13)

###
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)
    model = linear_model.LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    adjRsq=1-((1-score)*(X_test_rfe.shape[0]-1)/(X_test_rfe.shape[0]-n-1))
    print('nof:{} - Rsq:{:.4f} - adjRsq:{:.4f}'.format(n,score,adjRsq))
    if(adjRsq>r_sq):
        r_sq = adjRsq
        nof = nof_list[n]
        temp = pd.Series(rfe.support_,index = listnames)
        selected_features_rfe = temp[temp==True].index
        print(selected_features_rfe)

print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, r_sq))

'''
### FIT DATA
#listnames = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
listnames = ['CRIM', 'ZN', 'AGE', 'DIS', 'RAD', 'LSTAT']
X = dataframe[listnames]
Y = dataframe['MEDV']

### CREATE FITTED MODEL
lm=sm.OLS(Y,X).fit()
### PRINT SUMMARY
print(lm.summary())
#print(lm.condition_number)  #!!!!!
#print(lm.pvalues.values)
#rr =[i for i in lm.pvalues]
#for ix in range(len(listnames)):
    #print('{}:{:.4f}'.format(listnames[ix],rr[ix]))
### make prediction
#Y_pred=lm.predict(X)

### clear residual

resid = lm.resid_pearson
#print(resid[abs(resid)>3])
res_ser = pd.Series(resid**2,index = Y.index.values)
filt_res= res_ser[abs(res_ser)>7.5]
filt_res=filt_res.index.values
dataframe.drop([ix for ix in filt_res], inplace = True)
print(filt_res)

X = dataframe[listnames]
Y = dataframe['MEDV']

### CREATE FITTED MODEL
lm=sm.OLS(Y,X).fit()

### PRINT SUMMARY
print(lm.summary())  # check p-values


####  Histogram of standardized deviance residuals
plt.close('all')
resid = lm.resid
fig, ax = plt.subplots()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals')
fig.savefig('Histogram.png',dpi=125)

#### RESUDUALS PLOT
plt.close('all')
resid = lm.resid

#probplot = sm.ProbPlot(res)
fig = sm.qqplot(resid,stats.t, fit=True, line='45')
plt.title('Ex. 1 - qqplot - residuals of OLS fit')
plt.savefig('QQplot.png',dpi=125)

#####  Regression Diagnostics  ######
#### Outliers
plt.close('all')
sm.graphics.plot_leverage_resid2(lm)
plt.savefig('Outliers.png',dpi=125)

### Outlier test
outlier_test=lm.outlier_test()
#print('Outlier test (bonf(p)<0.05):')
#print(outlier_test[outlier_test['bonf(p)']<0.05])
##### Homoscedasticity

plt.close('all')
plt.plot(lm.resid,'o')
plt.title('Resudual plot')
plt.ylabel('Residual')
plt.xlabel('Observation numers')
plt.hist(lm.resid,density=True)
plt.savefig('residuals_plot.png',dpi=125)


print('~~~~~~~~~')