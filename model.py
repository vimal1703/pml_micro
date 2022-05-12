#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


# In[10]:


df = pd.read_csv('pre_process_students_performance.csv')


# In[11]:


X = df.drop(['performance'], axis = 1)
y = df['performance']


# In[12]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.25, random_state = 42)


# In[13]:


rfc = RandomForestClassifier(n_estimators = 10)
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)


# In[14]:


pickle.dump(rfc,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


# In[ ]:




