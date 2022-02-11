#!/usr/bin/env python
# coding: utf-8

# In[1]:



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


# In[2]:


toyStoryRatings = movieRatings['Toy Story (1995)']

# Correlamos el resto de peliculas (columnas) con la seleccionada (toy story)  
similarMovies = movieRatings.corrwith(toyStoryRatings)  
similarMovies = similarMovies.dropna()  
df = pd.DataFrame(similarMovies)

# Las ordenamos por el valor de score que hemos generado, de forma descendente  
similarMovies.sort_values(ascending=False)  


# In[14]:


import numpy as np  
# agregamos por titulo y devolvemos el  
# numero de veces que se puntuo, y la media de puntuacion  
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
movieStats
# nos quedamos con todas las que tengan mas de 100 puntuaciones  
# de distintos usuarios  
popularMovies = movieStats['rating']['size'] >= 100
# ordenamos por la puntuación asignada  
movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15]  


# In[15]:


# hacemos el join  
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))

# Ordenamos el dataframe por similaridad, y vemos los primeros 15 resultados  
df.sort_values(['similarity'], ascending=False)[:15]  


# In[ ]:


## Data Preparation and Modeling

- Standardize the data.
- Learn a Linear Discriminant Analysis model.

import pandas as pd
train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')
train.dtypes

X = train.drop('Survived', axis=1)
y = train['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train.head(10)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_features = train.select_dtypes(include=['int64', 'float64']).drop(['Survived'], axis=1).columns
numeric_features
categorical_features = train.select_dtypes(include=['object']).columns
categorical_features

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define model
from sklearn.ensemble import RandomForestClassifier
rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

# Preprocessing of training data, fit model 
rf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = rf.predict(X_test)
preds

print('MAE:', mean_absolute_error(y_test, preds))

