#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import sklearn


# In[3]:


df_data = pd.read_csv("dataset_unidad3-4.csv")

#print(df_link.head())
print(df_data.shape)

#explorando los datos
df_data.info()


# Graficas

# In[7]:


plt.hist(df_data.Sex)


# In[6]:


plt.hist(df_data.groupby(["Cabin"])['PassengerId'].count())


# In[8]:


df_data.groupby(["Cabin"])["Age"].count()


# In[9]:


df_data.mean()


# In[10]:


df_data.rename(str.lower, axis='columns')


# In[ ]:




