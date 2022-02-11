#!/usr/bin/env python
# coding: utf-8

# # Clase 6 final

# - Selecciona una columna del dataset final y aplica a los datos alguna operación matemática como por ejemplo pow(), exp(), etc.. . En el caso de que la función elegida admita 2 parametros utilizar una constante por ejemplo Pi, Tau, etc...

# In[2]:


import math
import pandas as pd
import numpy as np
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


# In[3]:


movieRatings['Toy Story (1995)']


# In[19]:


df_test = movieRatings.to_numpy()


# In[20]:


df = movieRatings.fillna(0)


# In[21]:


columna_seleccionada = df['1-900 (1994)']


# In[22]:


type(columna_seleccionada)


# In[23]:


columna_seleccionada_np = columna_seleccionada.to_numpy()


# In[24]:


type(columna_seleccionada_np)


# In[25]:


df.head()


# In[26]:


import math


# In[39]:


columna_seleccionada_pow=[math.pow(i, math.pi) for i in columna_seleccionada_np]


# In[52]:


columna_seleccionada_pow


# In[53]:


columna_seleccionada_exp=[math.exp(i) for i in columna_seleccionada_np]


# In[54]:


columna_seleccionada_exp


# In[62]:


columna_seleccionada_trunc=[ math.trunc(i) for i in columna_seleccionada_np]


# In[61]:


columna_seleccionada_trunc


# In[66]:


columna_seleccionada_fmod=[ math.fmod(i,math.e) for i in columna_seleccionada_np]


# In[67]:


columna_seleccionada_fmod


# In[68]:


columna_seleccionada_close=[ math.isclose(i,math.tau) for i in columna_seleccionada_np]


# In[69]:


columna_seleccionada_close


# In[ ]:




