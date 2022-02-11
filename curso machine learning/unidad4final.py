#!/usr/bin/env python
# coding: utf-8

# # Composición de filas y columnas del dataset.
# # Descripción de los campos.

# In[2]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import sklearn

df_link = pd.read_csv('./mllinks.csv')
df_movies = pd.read_csv("./movies.csv")
df_ratings = pd.read_csv("./ratings.csv")

df_tags = pd.read_csv("./tags.csv")

#print(df_link.head())
print(df_link.shape)

#explorando los datos
df_link.info()


# 
# Los identificadores que se pueden usar para enlazar con otras fuentes de datos de la película están contenidos en el archivo "links.csv". Cada línea de este archivo después de la fila de encabezamiento representa una película, y tiene el siguiente formato:
# 
#     movieId,imdbId,tmdbId
# 
# movieId es un identificador de películas usado por <https://movielens.org>. Por ejemplo, la película Toy Story tiene el enlace <https://movielens.org/movies/1>.
# 
# imdbId es un identificador para las películas usadas por <http://www.imdb.com>. Por ejemplo, la película Toy Story tiene el enlace <http://www.imdb.com/title/tt0114709/>.
# 
# tmdbId es un identificador para las películas usadas por <https://www.themoviedb.org>. Por ejemplo, la película Toy Story tiene el enlace <https://www.themoviedb.org/movie/862>.
# 

# In[1]:


print(df_movies.head())
print(df_movies.shape)

#explorando los datos
df_movies.info()


# La información de la película está contenida en el archivo "movies.csv". Cada línea de este archivo después de la fila de encabezamiento representa una película, y tiene el siguiente formato:
# 
#     movieId,título,géneros
# 
# Los títulos de las películas se introducen manualmente o se importan desde <https://www.themoviedb.org/>, e incluyen el año de estreno entre paréntesis. Pueden existir errores e inconsistencias en estos títulos.
# 
# Los géneros son una lista separada de la tubería, y se seleccionan de los siguientes:
# 
# * Acción
# * Aventura
# * Animación
# * Niños
# * Comedia
# * Crimen
# * Documental
# * Drama
# * Fantasía
# * Film-Noir
# * Horror
# * Musical
# * Misterio
# * Romance
# * Ciencia Ficción
# * Thriller
# * Guerra
# * Occidental
# * (no hay géneros listados)
# 

# In[59]:


print(df_ratings.head())

print(df_ratings.shape)

#explorando los datos
df_ratings.info()


# Todas las clasificaciones están contenidas en el archivo "ratings.csv". Cada línea de este archivo después de la fila de encabezamiento representa una calificación de una película por un usuario, y tiene el siguiente formato:
# 
#     userId,movieId,rating,timestamping
# 
# Las líneas de este archivo están ordenadas primero por el ID de usuario, luego, dentro del usuario, por el ID de la película.
# 
# Las clasificaciones se hacen en una escala de 5 estrellas, con incrementos de media estrella (0,5 estrellas - 5,0 estrellas).
# 
# Las marcas de tiempo representan segundos desde la medianoche del 1 de enero de 1970.

# In[ ]:


print(df_tags.head())
print(df_tags.shape)

#explorando los datos
df_tags.info() 

df_tags.describe()


# Todas las etiquetas están contenidas en el archivo `tags.csv`. Cada línea de este archivo después de la línea de encabezado representa una etiqueta aplicada a una película por un usuario, y tiene el siguiente formato:
# 
#     userId,movieId,tag,timestamp
# 
# Las líneas de este archivo están ordenadas primero por el ID de usuario, luego, dentro del usuario, por el ID de la película.
# 
# Las etiquetas son metadatos generados por el usuario sobre las películas. Cada etiqueta es típicamente una sola palabra o frase corta. El significado, el valor y el propósito de una etiqueta en particular es determinado por cada usuario.
# 
# Las marcas de tiempo representan segundos desde la medianoche del 1 de enero de 1970.
# 

# La mayoria de valoraciones esta entre 3 y 4 estrellas

# # Comentar como relacionar los datos de las diferentes tablas.
# 

# In[49]:


pd.merge(df_movies,df_ratings,how='inner',on='movieId').head(10)


# In[96]:


pd.merge(df_movies,df_tags,how='inner',on='movieId').head()


# # Valores perdidos y outlier.
# 

# In[99]:


total = df_movies.isnull().sum().sort_values(ascending=False)
percent_1 = df_movies.isnull().sum()/df_movies.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)


# In[100]:


total = df_link.isnull().sum().sort_values(ascending=False)
percent_1 = df_link.isnull().sum()/df_link.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)


# In[101]:



total = df_ratings.isnull().sum().sort_values(ascending=False)
percent_1 = df_ratings.isnull().sum()/df_ratings.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)


# In[102]:



total = df_tags.isnull().sum().sort_values(ascending=False)
percent_1 = df_tags.isnull().sum()/df_tags.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)


# # Alguna otra información que veáis relevante.

# In[78]:



n_users = df_ratings.userId.unique().shape[0]
n_items = df_ratings.movieId.unique().shape[0]
print (str(n_users) + ' usuarios')
print (str(n_items) + ' peliculas')


# In[97]:


agrupacion_tag = df_tags.groupby(["tag"])["movieId"].count()
agrupacion_tag.head()


# In[98]:


agrupacion_tag.describe()


# In[93]:


df_ratings.groupby(["rating"])["userId"].count()


# In[126]:


df_movies.genres.unique().shape[0]


# # Aportar gráficas para entender los datos.

# In[95]:


plt.hist(df_ratings.rating)


# In[136]:


plt.hist(df_tags.groupby(["movieId"])['tag'].count())


# In[ ]:




