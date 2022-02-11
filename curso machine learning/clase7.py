#!/usr/bin/env python
# coding: utf-8

# ### Crea una tubería (pipeline) que realice las siguientes tareas:
# 
# - Imputar valores perdidos
# - Escalar los valores
# - Transformar los valores categóricos en enteros
# - Aplicar un algoritmo para entrenar y predecir el resultado.
# 
# Se utilizará el dataset del Titanic

# In[44]:


#Importamos las librerias
import pandas as pd
import numpy as np
import math
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
## Data Preparation and Modeling
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# In[54]:


#leemos los csv
df_test = pd.read_csv("titanic_test.csv")
df_train=pd.read_csv("titanic_train.csv").fillna(0)
#mostramos las tablas
train.dtypes
df_train.head()


# In[46]:


df_train.info()


# In[57]:


X = train.drop('Survived', axis=1)
y = train['Survived']
# Preprocesamos los datos
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocesamos en datos categoricos
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_features = train.select_dtypes(include=['int64', 'float64']).drop(['Survived'], axis=1).columns
numeric_features
categorical_features = train.select_dtypes(include=['object']).columns
categorical_features

# Preprocesamiento  para datos numéricos y categóricos
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# definir modelo
from sklearn.ensemble import RandomForestClassifier
rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

# Preprocessing of training data, fit model 
rf.fit(X_train, y_train)

# Preprocesamiento de validación de los datos , obtenemos las predicciones


preds = rf.predict(X_test)
preds

print('MAE:', mean_absolute_error(y_test, preds))


# In[58]:


y_pred = rf.predict(X_test)
y_pred 


# In[59]:


X_test


# In[ ]:




