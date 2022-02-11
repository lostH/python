#!/usr/bin/env python
# coding: utf-8

# In[4]:



import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']  
ratings = pd.read_csv("ml-100k/u.data", sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']  
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

# combinamos ambos datasets para tener el  
ratings = pd.merge(movies, ratings)

# Pivotamos la tabla para que la matriz tenga : fila por usuario y columna por pelicula
movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')  
movieRatings.head()
#limpianos los nan


# In[27]:


movieRatings.isnull().sum()
movieRatings.dropna(inplace=True)
movieRatings.isnull().sum() 


# In[28]:



movieRatings.head()


# In[ ]:




