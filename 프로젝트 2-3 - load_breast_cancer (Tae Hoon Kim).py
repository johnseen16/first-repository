#!/usr/bin/env python
# coding: utf-8

# In[36]:


pip install scikit-learn  


# In[37]:


pip install matplotlib


# In[38]:


from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()

print(dir(breast_cancer))
# dir()는 객체가 어떤 변수와 메서드를 가지고 있는지 나열함


# In[39]:


breast_cancer.keys()


# In[40]:


breast_cancer_data = breast_cancer.data

print(iris_data.shape) 
#shape는 배열의 형상정보를 출력


# In[41]:


breast_cancer_data[0]


# In[42]:


breast_cancer_label = breast_cancer.target
print(breast_cancer_label.shape)
breast_cancer_label


# In[43]:


breast_cancer.target_names


# In[45]:


print(breast_cancer.DESCR)


# In[46]:


breast_cancer.feature_names


# In[47]:


breast_cancer.filename


# In[48]:


import pandas as pd

print(pd.__version__)


# In[49]:


breast_cancer_df = pd.DataFrame(data=breast_cancer_data, columns=breast_cancer.feature_names)
breast_cancer_df


# In[50]:


breast_cancer_df["label"] = breast_cancer.target
breast_cancer_df


# In[51]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data, 
                                                    breast_cancer_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

print('X_train 개수: ', len(X_train),', X_test 개수: ', len(X_test))


# In[52]:


X_train.shape, y_train.shape


# In[53]:


X_test.shape, y_test.shape


# In[54]:


y_train, y_test


# In[55]:


#Decision Tree Classifier for the X_train and the Y_Train
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)
print(decision_tree._estimator_type)


# In[56]:


decision_tree.fit(X_train, y_train)


# In[58]:


y_pred = decision_tree.predict(X_test)
y_pred


# In[59]:


y_test


# In[60]:


#The accuracy score of the Y_test and the Y_pred (Y Prediction)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[61]:


# (1) 필요한 모듈 import (breast_cancer)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# (2) 데이터 준비 (breast_cancer datatset loading)
breast_cancer = load_breast_cancer()
breast_cancer_data = breast_cancer.data
breast_cancer_label = breast_cancer.target

# (3) train, test 데이터 분리 (여기서 breast_cancer dataset에 데이터 분리 와 분석을 의미한다.)
X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data, 
                                                    breast_cancer_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

# (4) 모델 학습 및 예측 (decision_tree을 이용해 모델을 학습 하며 점수를 예측할수있다.)
decision_tree = DecisionTreeClassifier(random_state=32)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

print(classification_report(y_test, y_pred))


# In[62]:


from sklearn.ensemble import RandomForestClassifier 

X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data, 
                                                    breast_cancer_label, 
                                                    test_size=0.2, 
                                                    random_state=21)

random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))


# In[63]:


from sklearn import svm
svm_model = svm.SVC()

print(svm_model._estimator_type)


# In[64]:


svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[65]:


from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier()

print(sgd_model._estimator_type)


# In[66]:


sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[67]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()

print(logistic_model._estimator_type)


# In[68]:


logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[ ]:




