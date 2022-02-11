#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Imports necesarios
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[6]:



#cargamos los datos de entrada
data = pd.read_csv("./dataset_unidad3-4.csv")
#veamos cuantas dimensiones y registros contiene
data.shape


# In[7]:


#son 161 registros con 8 columnas. Veamos los primeros registros
data.head()


# In[8]:


data.describe()


# In[9]:



# Visualizamos rápidamente las caraterísticas de entrada
data.drop(['PassengerId','Survived', 'Age'],1).hist()
plt.show()


# In[18]:


filtered_data = data[(data['PassengerId'] <= 3500) & (data['Survived'] <= 891)]
 
colores=['orange','blue']
tamanios=[30,60]
 
f1 = filtered_data['PassengerId'].values
f2 = filtered_data['Age'].values
 
# Vamos a pintar en colores los puntos por debajo y por encima de la media de Cantidad de Palabras
asignar=[]
for index, row in filtered_data.iterrows():
    if(row['Survived']>0):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
    
plt.scatter(f1, f2, c=asignar, s=tamanios[0])
plt.show()


# In[19]:


# Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
dataX =filtered_data[["PassengerId"]]
X_train = np.array(dataX)
y_train = filtered_data['Survived'].values

# Creamos el objeto de Regresión Linear
regr = linear_model.LinearRegression()

# Entrenamos nuestro modelo
regr.fit(X_train, y_train)

# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred = regr.predict(X_train)

# Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
print('Coefficients: \n', regr.coef_)
# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: \n', regr.intercept_)
# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' % r2_score(y_train, y_pred))


# In[20]:



#Vamos a comprobar:
# Quiero predecir cuántos "Shares" voy a obtener por un artículo con 2.000 palabras,
# según nuestro modelo, hacemos:
y_Dosmil = regr.predict([[2000]])
print(int(y_Dosmil))


# In[ ]:




