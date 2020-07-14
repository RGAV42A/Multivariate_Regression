import pandas as pd
import numpy as np
#from pylab import *
import matplotlib.pyplot as plt
#from tempfile import TemporaryFile
import random,lxml,xlrd,sys,json
#from lxml.html import parse
import urllib,re,requests
#from lxml import objectify
#from pandas.io.json import json_normalize
#from pandas.io.json import loads
#from pandas.io.pytables import HDFStore
import pickle,random
#from sqlalchemy import create_engine
from numpy import nan as NA
#from pandas_datareader import data as web # da instaliram na moja komp
# import pandas.io.data as web
#import fix_yahoo_finance # lipsva
import datetime,sqlite3
#import pandas.io.sql as sql
#from sklearn.neural_network import *
#from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
#from sklearn import preprocessing
##from sklearn import linear_model,feature_selection
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.pipeline import make_pipeline
import statsmodels.graphics.api as smg
from statsmodels.stats.outliers_influence import variance_inflation_factor,OLSInfluence
import statsmodels.api as sm
import statsmodels.stats.diagnostic as ssd
##from scipy.optimize import curve_fit
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist, pdist

###########################################################

### Multivariate Regression  #####
'''
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
##### read data
dataframe=pd.read_csv('/home/ragav/my_python_files/folder_data/BostonHausing.csv')
#dataframe=pd.read_csv('/home/ragav/my_python_files/folder_data/BostonHausing_short.csv')

##### create correlation matix
corr=dataframe.corr()
###### plot coor matrix
smg.plot_corr(corr,xnames=list(corr.columns))
plt.savefig('fig1.png',dpi=125)

###### remove multicollinearity
## create list of features
listnames = ['CRIM', 'ZN', 'AGE', 'DIS', 'RAD', 'LSTAT']
## calc VIF
vif = [variance_inflation_factor(dataframe[listnames].values, ix) for ix in range(dataframe[listnames].shape[1])]
for i in range(len(vif)):
    print('vif for {} = {:.1f}'.format(listnames[i],vif[i]))

#######   Build the Multivariate Linear Regression Model
### list with features
listnames = ['CRIM', 'ZN', 'AGE', 'DIS', 'RAD', 'LSTAT']
#listnames = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
### make boxplot
#dataframe[listnames].boxplot()
#plt.savefig('fig.png',dpi=125)
### sort features from dataframe
X = dataframe[listnames]
Y = dataframe['MEDV']
### create fitted model
lm=sm.OLS(Y,X).fit()
### print the summary
print(lm.summary())
### make prediction
#Y_pred=lm.predict(X)
#### residuals plot
res = lm.resid
probplot = sm.ProbPlot(res)
fig2 = probplot.qqplot()
h = plt.title('Ex. 1 - qqplot - residuals of OLS fit')
plt.savefig('fig2.png',dpi=125)

#### normality test
nameindex = list(dataframe.columns)
for ix in nameindex:
    p_val=ssd.kstest_normal(dataframe[ix].values)
    print('p-value for {} = {}'.format(ix,p_val))

#####  Regression Diagnostics  ######
#### Outliers
sm.graphics.plot_leverage_resid2(lm)
plt.savefig('fig3.png',dpi=125)
### Outlier test
outlier_test=lm.outlier_test()
print('Outlier test (bonf(p)<0.05):')
print(outlier_test[outlier_test['bonf(p)']<0.05])
##### Homoscedasticity
plt.plot(lm.resid,'o')
plt.title('Resudual plot')
plt.ylabel('Residual')
plt.xlabel('Observation numers')
plt.hist(lm.resid,density=True)
plt.savefig('fig4.png',dpi=125)
