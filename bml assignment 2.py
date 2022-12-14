#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[5]:


data=[[43,99],[21,65],[25,79],[42,74],[57,87],[59,81]]


# In[24]:


ds=pd.DataFrame(data, columns=['AGE','GLUCOSE LEVEL'])
ds.set_index(np.array(range(1,7)))


# In[8]:


x=np.array(ds['AGE'])
y=np.array(ds['GLUCOSE LEVEL'])
print(y)
print(x)


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


plt.scatter(x,y)
plt.xlabel('AGE')
plt.ylabel('GLUCOSE LEVEL')


# In[15]:


#object for regression
x=np.array(x.reshape((-1, 1)))
reg = linear_model.LinearRegression()
reg.fit(x,y)


# In[16]:


reg.predict(np.array([55]).reshape(-1, 1))


# In[17]:


reg.coef_   #m


# In[21]:


reg.intercept_   #c


# In[22]:


(0.38455339*55)+65.0025520483546


# In[19]:


#y=m*x+b which is the predicted value
#(135.78767123*3300)+180616.43835616432 #hw-sum


# In[20]:


#(135.78767123*5000)+180616.43835616432  hw-sum


# In[ ]:




