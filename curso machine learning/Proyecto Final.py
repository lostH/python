#!/usr/bin/env python
# coding: utf-8

# ### Proyecto final

# In[1]:


#Cargamos las librerías que utilizaremos
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import sklearn


# In[91]:


r_cols = ['user_id', 'movie_id', 'rating']  
ratings = pd.read_csv("ml-100k/u.data", sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']  
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

# combinamos ambos datasets para tener el  
df_ratings = pd.merge(movies, ratings)

 

#limpianos los nan
#previsualizamos


# In[92]:


df_ratings


# In[93]:


n_users = df_ratings.user_id.unique().shape[0]
n_items = df_ratings.title.unique().shape[0]
print (str(n_users) + ' users')
print (str(n_items) + ' items')


# Vemos que hay 934 usuarios y 1664 titulos valorados

# In[94]:


plt.hist(df_ratings.rating,bins=8)


# tenemos 5000 valoraciones de 1 y 35000 de 4

# In[95]:


df_ratings.groupby(["rating"])["user_id"].count()


# aqui estan las valoraciones exactas

# In[104]:


# Pivotamos la tabla para que la matriz tenga : fila por usuario y columna por pelicula
movieRatings = pd.pivot_table( df_ratings, values='rating', index=['user_id'], columns=['title']).fillna(0) 
movieRatings


# In[105]:


#porcentajes
Ratings=movieRatings.values
sparsity = float(len(Ratings.nonzero()[0]))
sparsity /= (Ratings2.shape[0] * Ratings.shape[1])
sparsity *= 100
print('Sparsity: {:4.2f}%'.format(sparsity))


# Dividimos en Train y Test set Separamos en train y test para -más adelante- poder medir la calidad de nuestras recomendaciones.

# In[112]:


Ratings_train, Ratings_test = train_test_split(Ratings, test_size = 0.2, shuffle=False, random_state=42)


# In[113]:


Ratings_train.shape


# In[114]:


Ratings_test.shape


# - Matriz de similitud entre los usuarios (distancia del coseno -vectores-).
# - Predecir la valoración desconocida de un ítem i para un usuario activo u basandonos en la suma ponderada de todas las valoraciones del resto de usuarios para dicho ítem.
# - Recomendaremos los nuevos ítems a los usuarios según lo establecido en los pasos anteriores.

# In[117]:


sim_matrix = 1 - sklearn.metrics.pairwise.cosine_distances(Ratings)


# In[118]:


sim_matrix.shape


# In[119]:


sim_matrix


# plt.rcParams['figure.figsize'] = (20.0, 5.0)
# plt.imshow(sim_matrix);
# plt.colorbar()
# plt.show()

# In[121]:


#separar las filas y columnas de train y test
sim_matrix_train = sim_matrix[0:754,0:754]
sim_matrix_test = sim_matrix[754:943,754:943]
print(sim_matrix_train.shape)
print(sim_matrix_test.shape)


# Predicciones (las recomendaciones!)

# In[122]:


users_predictions = sim_matrix_train.dot(Ratings_train) / np.array([np.abs(sim_matrix_train).sum(axis=1)]).T


# In[123]:


users_predictions.shape


# In[124]:


plt.rcParams['figure.figsize'] = (20.0, 5.0)
plt.imshow(users_predictions);
plt.colorbar()
plt.show()


# Vemos que hay algunas recomendaciones que estan cerca del 2 al 2,5

# ## Ejemplo

# In[155]:


#abrimos el dataset de usuarios
u_cols = ['user_id', 'age', 'gender','occupation','zip code']  

users = pd.read_csv("ml-100k/u.user", sep='|', names=u_cols, usecols=range(5), encoding="ISO-8859-1")
users


# In[180]:


USUARIO_EJEMPLO = '15213' 	 # debe existir en nuestro dataset de train!
data = users[users['zip code'] == USUARIO_EJEMPLO]
usuario_ver = data.iloc[0]['user_id'] -1 # resta 1 para obtener el index de pandas
user0=users_predictions.argsort()[usuario_ver]

# Veamos los tres recomendados con mayor puntaje en la predic para este usuario
for i, aMov in enumerate(user0[-3:]):
    selMo = df_ratings[df_ratings['movie_id']==(aMov+1)]
    print(selMo['title'] , 'puntaje:', users_predictions[usuario_ver][aMov])


# Las recomendaciones que da para el usuario son
# - Amos & Andrew (1993) con puntaje: 3.0472330279409636
# - Big Bang Theory, The (1994)con puntaje: 3.1624639802354033
# - Stranger in the House (1997)con puntaje: 3.784816722512285

# In[182]:


def get_mse(preds, actuals):
    if preds.shape[1] != actuals.shape[1]:
        actuals = actuals.T
    preds = preds[actuals.nonzero()].flatten()
    actuals = actuals[actuals.nonzero()].flatten()
    return mean_squared_error(preds, actuals)


# In[183]:


get_mse(users_predictions, Ratings_train)


# In[185]:


# Realizo las predicciones para el test set
users_predictions_test = sim_matrix.dot(Ratings) / np.array([np.abs(sim_matrix).sum(axis=1)]).T
users_predictions_test = users_predictions_test[754:943,:]

get_mse(users_predictions_test, Ratings_test)


# In[ ]:




