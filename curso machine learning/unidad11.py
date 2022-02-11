#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Create your first MLP in Keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy

#cargamos datos

FILENAME = './pima-indians-diabetes.csv'

# se inicia el generador aleatorio con la variable seed, para que siempre obtengamos los mismos resultados.
seed = 7
numpy.random.seed(seed)

# cargamos Pima indians dataset.
dataset = numpy.loadtxt(FILENAME, delimiter=',')
# hacemos los trainings
training_data = dataset[:, 0:8]
training_targets = dataset[:, 8]

# Creacion del modelo
#utilizaremos una red conectada completamente con 3 capas.
#las Capas estaran completamente conectadas y son definidas mediante la clase Dense. 
#En las primeras dos capas se definirá una función de activación relu 
#y para la capa de salida una función sigmoid. En el pasado las funciones sigmoid  
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training_data, training_targets, epochs=150, batch_size=10)
scores = model.evaluate(training_data, training_targets)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


# In[ ]:




