#!/usr/bin/env python
# coding: utf-8

# In[19]:


pip install scikit-learn  


# In[20]:


pip install matplotlib


# In[21]:


from sklearn.datasets import load_digits

digits = load_digits()

print(dir(digits))
# dir()는 객체가 어떤 변수와 메서드를 가지고 있는지 나열함


# In[22]:


digits.keys()


# In[23]:


digits_data = digits.data

print(digits_data.shape) 
#shape는 배열의 형상정보를 출력


# In[24]:


digits_data[0]


# In[25]:


digits_label = digits.target
print(digits_label.shape)
digits_label


# In[26]:


digits.target_names


# In[16]:


print(digits.DESCR)


# In[27]:


digits.feature_names


# In[53]:


digits.filename


# In[33]:


import pandas as pd

print(pd.__version__)


# In[34]:


digits_df = pd.DataFrame(data=digits_data, columns=digits.feature_names)
digits_df


# In[35]:


digits_df["label"] = digits.target
digits_df


# In[36]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(digits_data, 
                                                    digits_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

print('X_train 개수: ', len(X_train),', X_test 개수: ', len(X_test))


# In[37]:


X_train.shape, y_train.shape


# In[38]:


X_test.shape, y_test.shape


# In[39]:


y_train, y_test


# In[40]:


#Decision Tree Classifier for the X_train and the Y_Train
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)
print(decision_tree._estimator_type)


# In[41]:


decision_tree.fit(X_train, y_train)


# In[42]:


y_pred = decision_tree.predict(X_test)
y_pred


# In[43]:


y_test


# In[44]:


#The accuracy score of the Y_test and the Y_pred (Y Prediction)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[45]:


# (1) 필요한 모듈 import (digits)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# (2) 데이터 준비 (digits datatset loading)
digits = load_digits()
digits_data = digits.data
digits_label = digits.target

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


# In[46]:


from sklearn.ensemble import RandomForestClassifier 

X_train, X_test, y_train, y_test = train_test_split(digits_data, 
                                                    digits_label, 
                                                    test_size=0.2, 
                                                    random_state=21)

random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))


# In[47]:


from sklearn import svm
svm_model = svm.SVC()

print(svm_model._estimator_type)


# In[48]:


svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[49]:


from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier()

print(sgd_model._estimator_type)


# In[50]:


sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[51]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()

print(logistic_model._estimator_type)


# In[52]:


logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[ ]:




