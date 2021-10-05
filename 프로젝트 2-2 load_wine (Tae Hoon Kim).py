#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install scikit-learn  


# In[2]:


pip install matplotlib


# In[3]:


from sklearn.datasets import load_wine

wine = load_wine()

print(dir(wine))
# dir()는 객체가 어떤 변수와 메서드를 가지고 있는지 나열함


# In[4]:


wine.keys()


# In[5]:


wine_data = wine.data

print(wine_data.shape) 
#shape는 배열의 형상정보를 출력


# In[6]:


wine_data[0]


# In[7]:


wine_label = wine.target
print(wine_label.shape)
wine_label


# In[8]:


wine.target_names


# In[16]:


print(digits.DESCR)


# In[10]:


wine.feature_names


# In[11]:


wine.filename


# In[12]:


import pandas as pd

print(pd.__version__)


# In[15]:


wine_df = pd.DataFrame(data=wine_data, columns=wine.feature_names)
wine_df


# In[16]:


wine_df["label"] = wine.target
wine_df


# In[17]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(wine_data, 
                                                    wine_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

print('X_train 개수: ', len(X_train),', X_test 개수: ', len(X_test))


# In[18]:


X_train.shape, y_train.shape


# In[19]:


X_test.shape, y_test.shape


# In[20]:


y_train, y_test


# In[21]:


#Decision Tree Classifier for the X_train and the Y_Train
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)
print(decision_tree._estimator_type)


# In[22]:


decision_tree.fit(X_train, y_train)


# In[23]:


y_pred = decision_tree.predict(X_test)
y_pred


# In[24]:


y_test


# In[25]:


#The accuracy score of the Y_test and the Y_pred (Y Prediction)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[27]:


# (1) 필요한 모듈 import (wine)
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# (2) 데이터 준비 (wine datatset loading)
digits = load_wine()
digits_data = wine.data
digits_label = wine.target

# (3) train, test 데이터 분리 (여기서 digits dataset에 데이터 분리 와 분석을 의미한다.)
X_train, X_test, y_train, y_test = train_test_split(digits_data, 
                                                    digits_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

# (4) 모델 학습 및 예측 (decision_tree을 이용해 모델을 학습 하며 점수를 예측할수있다.)
decision_tree = DecisionTreeClassifier(random_state=32)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

print(classification_report(y_test, y_pred))


# In[28]:


from sklearn.ensemble import RandomForestClassifier 

X_train, X_test, y_train, y_test = train_test_split(wine_data, 
                                                    wine_label, 
                                                    test_size=0.2, 
                                                    random_state=21)

random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))


# In[29]:


from sklearn import svm
svm_model = svm.SVC()

print(svm_model._estimator_type)


# In[30]:


svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[31]:


from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier()

print(sgd_model._estimator_type)


# In[32]:


sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[33]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()

print(logistic_model._estimator_type)


# In[52]:


logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[ ]:




