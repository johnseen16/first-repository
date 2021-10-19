#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import os
from os.path import join

import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import xgboost as xgb
import lightgbm as lgb
import scipy as sp

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import seaborn as sns

print('CONGRATS💩')


# In[2]:


#데이터 가저오기. kaggle에있는 데이터를 가져와서 test and train!
data_dir = os.getenv('HOME') + '/aiffel/kaggle_kakr_housing/data'

train_data_path = join(data_dir, 'train.csv')
test_data_path = join(data_dir, 'test.csv')

train = pd.read_csv(train_data_path)
test = pd.read_csv(test_data_path)


# In[3]:


train.head(4)


# In[4]:


test.head(4)


# In[5]:


cols = train.columns
cols


# In[6]:


#데이터 탐색하고 전처리! GO!
train.isnull().sum()


# In[7]:


print(len(train['id']))
print(len(pd.value_counts(train['id'])))


# In[8]:


fig, ax = plt.subplots(figsize=(7,5))
sns.distplot(train['price'])


# In[9]:


fig = plt.figure(figsize = (10,5))

fig.add_subplot(1,2,1)
res = sp.stats.probplot(train['price'], plot=plt)

fig.add_subplot(1,2,2)
res = sp.stats.probplot(np.log1p(train['price']), plot=plt)


# In[10]:


train['price'] = np.log1p(train['price'])
sns.kdeplot(train['price'])
plt.show()


# In[11]:


#변수시각화!!! 반응변수 (SPEARMAN!!!)
cor_abs = abs(train.corr(method='spearman'))
cor_cols = cor_abs.nlargest(n=10, columns='price').index # price과 correlation이 높은 column 10개 뽑기(내림차순)
cor = np.array(sp.stats.spearmanr(train[cor_cols].values))[0] # 10 x 10
print(cor_cols.values)
plt.figure(figsize=(10,10))
sns.set(font_scale=1.25)
sns.heatmap(cor, fmt='.2f', annot=True, square=True , annot_kws={'size' : 8} ,xticklabels=cor_cols.values, yticklabels=cor_cols.values)


# In[12]:


# GRADE GRADE
data = pd.concat([train['price'], train['grade']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='grade', y='price', data = data)


# In[13]:


train.loc[(train['grade'] == 3) & (train['price'] > 12)]


# In[14]:


train.loc[(train['grade'] == 7) & (train['price'] > 14.6)]


# In[15]:


train.loc[(train['grade'] == 8) & (train['price'] > 14.8)]


# In[16]:


train.loc[(train['grade'] == 11) & (train['price'] > 15.5)]


# In[17]:


train.loc[(train['grade'] == 3) & (train['price'] > 12)]


# In[18]:


train.loc[(train['grade'] == 7) & (train['price'] > 14.5)]


# In[19]:


train.loc[(train['grade'] == 8) & (train['price'] > 14.7)]


# In[21]:


train.loc[(train['grade'] == 11) & (train['price'] > 15.5)]


# In[22]:


train = train.loc[train['id'] != 2302]
train = train.loc[train['id'] != 4123]
train = train.loc[train['id'] != 12346]
train = train.loc[train['id'] != 7173]
train = train.loc[train['id'] != 2775]
print(len(train['id']))


# In[23]:


#sqft_above
data = pd.concat([train['price'], train['sqft_living']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.scatterplot(x='sqft_living', y='price', data = data)


# In[24]:


train.loc[train['sqft_living']>13000]


# In[25]:


train = train.loc[train['id'] != 8912]
print(len(train['id']))


# In[26]:


#sqft_living!
data = pd.concat([train['price'], train['sqft_above']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.regplot(x='sqft_above', y='price', data = data)


# In[ ]:


#bathrooms!!!
data = pd.concat([train['price'], train['bathrooms']], axis=1)
f, ax = plt.subplots(figsize=(18, 7))
fig = sns.boxplot(x='bathrooms', y='price', data = data)


# In[ ]:


#lat 위도
data = pd.concat([train['price'], train['lat']], axis=1)
f, ax = plt.subplots(figsize=(9, 7))
fig = sns.scatterplot(x='lat', y='price', data = data)


# In[29]:


#date에 형태를 변환 
train['date'] = train['date'].apply(lambda i: i[:6]).astype(int)
test['date'] = test['date'].apply(lambda i: i[:6]).astype(int)


# In[30]:


#floors, waterfront, condition!
print(train['waterfront'].unique())
print(train['view'].unique())
print(train['condition'].unique())


# In[31]:


#여기서 id test train! ERASE!
del train['id']
del test['id']


# In[32]:


fig, ax = plt.subplots(10, 2, figsize=(20, 60), constrained_layout=True)
columns = train.columns
count = 0
for row in range(10):
    for col in range(2):
        sns.kdeplot(train[columns[count]], ax=ax[row][col])
        ax[row][col].set_title(columns[count], fontsize=15)
        count += 1
        if count == 19 :
            break


# In[33]:


skew_columns = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']

for c in skew_columns:
    train[c] = np.log1p(train[c].values)
    test[c] = np.log1p(test[c].values)

fig, ax = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

count = 0
for row in range(2):
    for col in range(2):
        if count == 5:
            break
        sns.kdeplot(train[skew_columns[count]], ax=ax[row][col])
        ax[row][col].set_title(skew_columns[count], fontsize=15)
        count += 1


# In[34]:


#target data
y = train['price']
del train['price']


# In[35]:


train.info()


# In[36]:


#RMSE!
def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))


# In[37]:


def get_scores(models, train, y):
    df = {}
    for model in models:
        
        model_name = model.__class__.__name__
        
        X_train, X_test, y_train, y_test = train_test_split(train, y, random_state=random_state, test_size=0.2)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        df[model_name] = rmse(y_test, y_pred)
        
        score_df = pd.DataFrame(df, index=['RMSE']).T.sort_values('RMSE', ascending=False)
    return score_df


# In[38]:


random_state = 2020

gboost = GradientBoostingRegressor(random_state=random_state)
xgboost = XGBRegressor(random_state=random_state)
lightgbm = LGBMRegressor(random_state=random_state)
rdforest = RandomForestRegressor(random_state=random_state)

models = [gboost, xgboost, lightgbm, rdforest]

get_scores(models, train, y)


# In[ ]:


#하이퍼파라미터 탐색!!
def my_GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5):

    grid_model = GridSearchCV(model,
                              param_grid=param_grid,
                              scoring='neg_mean_squared_error',
                              cv=5,
                              verbose=verbose,
                              n_jobs=n_jobs)

    grid_model.fit(train, y)

    params = grid_model.cv_results_['params']          
    score = grid_model.cv_results_['mean_test_score']  

    results = pd.DataFrame(params)
    results['score'] = score

    results['RMSLE'] = np.sqrt(-1 * results['score'])  
    results = results.rename(columns={'RMSE': 'RMSLE'})
    results = results.sort_values('RMSLE')  
    return results


# In[40]:


# LGBM Regressor!
param_grid = {
    "n_estimators":[50, 100, 500],
    "max_depth":[1, 6, 10],
    "learning_rate": [0.05, 0.1],
    "boosting_type": ['gbdt', 'rf', 'dart', 'goss']
}

lgbmr_model = LGBMRegressor(random_state=random_state)

my_GridSearch(lgbmr_model, train, y, param_grid, verbose=2, n_jobs=5)


# In[ ]:


# XGBRegressor
param_grid = {
    "n_estimators":[50, 100, 500],
    "max_depth":[1, 6, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "booster": ['gblinear', 'gbtree', 'dart']
}

xgbr_model = XGBRegressor(random_state=random_state)

my_GridSearch(xgbr_model, train, y, param_grid, verbose=2, n_jobs=5)


# In[ ]:


# RandomForestRegressor
param_grid = {
    "n_estimators":[50, 100, 500],
    "max_depth":[1, 6, 10, None]
}

rf_model = RandomForestRegressor(random_state=random_state)

my_GridSearch(rf_model, train, y, param_grid, verbose=2, n_jobs=5)


# In[ ]:




