# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:38:04 2020

@author: sabab
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
#from lightgbm import LGBMRegressor

def performance(model, y_test, y_pred):
    model_name = str(model)
    model_name = model_name.split('(')
    model_name = model_name[0]
    df_temp=pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2'])
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)
    d =[ model_name, mae , mse, rmse, r2]
    df_temp = df_temp.append(pd.Series(d,index=['Model', 'MAE', 'MSE', 'RMSE', 'R2']),ignore_index=True)
    return df_temp

def comparePerformance(y_test, y_pred):        
    plt.figure(figsize=(15,8))
    plt.scatter(y_test,y_pred)
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.plot(y_test,label ='Test')
    plt.plot(y_pred, label = 'predict')
    plt.show()
    



warnings.filterwarnings('ignore')

train = pd.read_csv("train.csv",encoding='utf-8' ) 

df_final = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2'])
# =============================================================================
# Analyse the data 
# #sns.distplot(train['SalePrice'] , fit=norm);
# #
# ##Now plot the distribution
# #(mu, sigma) = norm.fit(train['SalePrice'])
# #plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
# #            loc='best')
# #plt.ylabel('Frequency')
# #plt.title('SalePrice distribution')
# #
# ##Get also the QQ-plot
# #fig = plt.figure()
# #res = stats.probplot(train['SalePrice'], plot=plt)
# #plt.show() 
# #
# #plt.figure(figsize=(30,8))
# #sns.heatmap(train.corr(),cmap='coolwarm',annot = True)
# #plt.show()
# 
# 
# #sns.lmplot(x='OverallQual',y='SalePrice',data=train)
# #sns.lmplot(x='GarageArea',y='SalePrice',data=train)
# #plt.scatter(x= 'GrLivArea', y='SalePrice', data = train)
# 
# #sns.boxplot(x='GarageCars',y='SalePrice',data=train)
# 
# #plt.figure(figsize=(16,8))
# #sns.boxplot(x='GarageCars',y='SalePrice',data=train)
# #plt.show()
# 
# =============================================================================

#Drop features which has more than 81 missing points
total = train.isnull().sum().sort_values(ascending=False)
percentage = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing = pd.concat([total, percentage], axis = 1, keys=['Total', 'Percentage'])
train = train.drop((missing[missing['Total']>81]).index, 1)


# Categorical boolean mask
categorical_feature_mask = train.dtypes==object
categorical_columns = list(train.columns[categorical_feature_mask])
labelEncoder = LabelEncoder()
train[categorical_columns] = train[categorical_columns].apply(lambda col: labelEncoder.fit_transform(col.astype(str)))

#check for the null values again
total_reduced = train.isnull().sum().sort_values(ascending=False)
print(train.isnull().sum().sort_values(ascending=False))
print("-"*80)

#replace the null values for the train columns
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())

#check for the null values in train again
print(train.isnull().sum().sort_values(ascending=False))
print("-"*80)


#pairwise correlation of the columns
n_features = 17 # how to features to select
cor = train.corr()
columns_selected = cor.nlargest(n_features, 'SalePrice')['SalePrice'].index

#=============================================================================
#plot to see correlation of the features
plt.figure(figsize=(16,8))
cm = np.corrcoef(train[columns_selected].values.T)
sns.set(font_scale=1.0)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=columns_selected.values, xticklabels=columns_selected.values)
plt.show()
#=============================================================================

#Change the train data to the selected featuers only
train = train[columns_selected]
total_selected = train.isnull().sum().sort_values(ascending=False)



X_train, X_test, y_train, y_test = train_test_split(train.drop('SalePrice', axis=1), train['SalePrice'], test_size=0.33, random_state=120)

#sclae the data using standard scalar
y_train= y_train.values.reshape(-1,1)
y_test= y_test.values.reshape(-1,1)
std_scaler_X = StandardScaler()
std_scaler_y = StandardScaler()
X_train = std_scaler_X.fit_transform(X_train)
X_test = std_scaler_X.fit_transform(X_test)
y_train = std_scaler_y.fit_transform(y_train)
y_test = std_scaler_y.fit_transform(y_test)


#linear regression
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
df_final = df_final.append(performance(model, y_test, y_pred))
#print(model.intercept_)
#print(model.coef_)
#see the plot of the train vs pred data
#comparePerformance(y_test, y_pred)

#decision tree regressor
p = {'random_state': 100}
model = DecisionTreeRegressor(**p)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
df_final = df_final.append(performance(model, y_test, y_pred))
#comparePerformance(y_test, y_pred)

#SVM regressor
p = {'kernel': 'rbf'}
model = SVR(**p)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
df_final = df_final.append(performance(model, y_test, y_pred))
#comparePerformance(y_test, y_pred)


#random forest regressor
p = {'n_estimators':500 ,'random_state': 100}
model = RandomForestRegressor(**p)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
df_final = df_final.append(performance(model, y_test, y_pred))
#comparePerformance(y_test, y_pred)


#adaptive boost regressor
p = { 'n_estimators': 500, 'learning_rate': 0.1, 'random_state': 0}
model = AdaBoostRegressor(**p)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
df_final = df_final.append(performance(model, y_test, y_pred))
#comparePerformance(y_test, y_pred)

#gradient boost regressor
p = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.02, 'loss': 'ls', 'random_state': 0}
model = GradientBoostingRegressor(**p) 
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
df_final = df_final.append(performance(model, y_test, y_pred))
#comparePerformance(y_test, y_pred)


#xtreme gradient boost regressor
p = {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.015, 'random_state': 12}
model = XGBRegressor(**p)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
df_final = df_final.append(performance(model, y_test, y_pred))
#comparePerformance(y_test, y_pred)


##light gradient boost regressor
#p = {'n_estimators': 500,'objective':'regression', 'num_leaves':5, 'learning_rate':0.1, 'max_bin':55, 'bagging_fraction':0.8} 
#model = LGBMRegressor(**p)
#model.fit(X_train,y_train)
#y_pred = model.predict(X_test)
#df_final = df_final.append(performance(model, y_test, y_pred))
##comparePerformance(y_test, y_pred)
#

df_final.to_csv('performance.csv',index=False)