#!/usr/bin/env python
# coding: utf-8

# El Titanic fue un buque de pasajeros británico que se hundió en el Océano Atlántico Norte en las primeras horas de la mañana del 15 de abril de 1912, después de que chocó con un iceberg durante su viaje inaugural desde Southampton a la ciudad de Nueva York. Hubo un estimado de 2,224 pasajeros y tripulantes a bordo del barco, y más de 1,500 murieron, convirtiéndolo en uno de los desastres marítimos comerciales más mortíferos en tiempos de paz en la historia moderna. El Titanic era el barco más grande a flote en el momento en que entró en servicio y fue el segundo de tres transatlánticos de clase olímpica operados por la White Star Line. El Titanic fue construido por el astillero Harland and Wolff en Belfast. Thomas Andrews, su arquitecto, murió en el desastre.

# In[18]:


#IMPORTANTE BIBLIOTECAS

# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# In[19]:


#cargando los datos
test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")


# In[20]:


#explorando los datos
train_df.info()

# El conjunto de entrenamiento tiene 891 ejemplos y 11 características + la variable objetivo (sobrevivió). 2 de los variables son float, 5 son enteros y 5 son objetos. A continuación he enumerado las características con una breve descripción:
# - Supervivencia: sobrevive o no 
# - Pasajeros: Identificación única de un pasajero. 
# - pclase:    Clase de billete     
# - Sexo:    Sexo     
# - Edad: Edad en años     
# - sibsp:    Número de hermanos/esposos a bordo
# 
# 
# - parch:  Número de padres e hijos a bordo    
# - tarifa:    Tarifa para pasajeros     
# - cabina:    Número de cabina     
# - embarcado:    Puerto de Embarque
# 

# In[21]:


train_df.describe()


# Arriba podemos ver que el 38% del conjunto de entrenamiento sobrevivió al Titanic. También podemos ver que la edad de los pasajeros va de 0,4 a 80 años. Además de eso, ya podemos detectar algunas características, que contienen valores perdidos, como la característica de la "Edad".

# In[22]:


train_df.head(8)


# De la tabla de arriba, podemos anotar algunas cosas. En primer lugar, que necesitamos convertir muchas características en numéricas más adelante, para que los algoritmos de aprendizaje de la máquina puedan procesarlas. Además, podemos ver que las características tienen rangos muy diferentes, que tendremos que convertir en aproximadamente la misma escala. También podemos detectar algunas características más, que contienen valores perdidos (NaN = no un número), que necesitamos tratar.

# In[23]:


total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)


# - La característica Embarcado tiene sólo 2 valores faltantes, que pueden ser rellenados fácilmente. 
# - Será mucho más difícil, tratar con la característica "Edad", que tiene 177 valores faltantes. 
# - La característica "Cabina" necesita una mayor investigación, pero parece que podríamos querer eliminarla del conjunto de datos, ya que falta el 77%.

# In[24]:



train_df.columns.values


# Arriba se pueden ver las 11 características + la variable objetivo (sobrevivió). ¿Qué características podrían contribuir a una alta tasa de supervivencia?
# Para mí tendría sentido  todo, excepto 'PassengerId', 'Ticket' y 'Name', ya que parecen tener una correlacion alta con la tasa de supervivencia.

# In[25]:


#EDAD Y SEXO

survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_df[train_df['Sex']=='female']
men = train_df[train_df['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')


# Se puede ver que los hombres tienen una alta probabilidad de supervivencia cuando tienen entre 18 y 30 años, lo que también es un poco cierto para las mujeres, pero no del todo. Para las mujeres las posibilidades de supervivencia son mayores entre los 14 y 40 años.
# 
# Para los hombres la probabilidad de supervivencia es muy baja entre los 5 y 18 años, pero no es así para las mujeres. Otra cosa que hay que tener en cuenta es que los bebés también tienen una probabilidad de supervivencia .
# 
# Como parece que hay ciertas edades que tienen mayores probabilidades de supervivencia y como quiero que todas las características estén aproximadamente en la misma escala, crearé grupos de edad más adelante.
# 

# In[26]:


#Embarcado, Clase y Sexo:

FacetGrid = sns.FacetGrid(train_df, row='Embarked', height=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()


# - Embarcado parece estar correlacionado con la supervivencia, dependiendo del género.
# 
# - Las mujeres en el puerto Q y en el puerto S tienen una mayor probabilidad de supervivencia. Pasa lo contrario si están en el puerto C. Los hombres tienen una alta probabilidad de supervivencia si están en el puerto C, pero una baja probabilidad si están en el puerto Q o S.
# 
# - La clase también parece estar correlacionada con la supervivencia. Generaremos otra gráfica de ello a continuación.

# In[27]:


#clase

sns.barplot(x='Pclass', y='Survived', data=train_df)


# Aquí vemos claramente, que la clase contribuye a la posibilidad de supervivencia de una persona, especialmente si esta persona está en la clase 1. Vamos a crear otra trama de Pclass a continuación.

# In[28]:


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass',height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# La trama anterior confirma nuestra suposición sobre la clase 1, pero también podemos detectar una alta probabilidad de que una persona de la clase 3 no sobreviva.

# SibSp y Parch tendría más sentido como una característica combinada, que muestra el número total de parientes, que una persona tiene en el Titanic. Lo crearé a continuación y también una característica que  si alguien no está solo.

# In[29]:


#SibSp and Parch:
data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
train_df['not_alone'].value_counts()


# In[30]:


axes = sns.catplot('relatives','Survived', 
                      data=train_df, aspect = 2.5, )


# Aquí podemos ver había una alta probabilidad de supervivencia con 1 a 3 parientes, pero una menor si tenía menos de 1 o más de 3 (excepto en algunos casos con 6 parientes).

# In[34]:


# preprocesamiento

#eliminamos id_pasajero ya que no es relevante

train_df = train_df.drop(['PassengerId'], axis=1)


# - valores perdidos
# 
# Tenemos que tratar con camarote (687), Embarcado (2) y Edad (177). Primero se penso de eliminar el camarote pero se detecto algo interesante. Un número de camarote se parece a 'C123' y la letra se refiere a la cubierta. Por lo tanto vamos a extraer estos y crear una nueva característica, que contiene la cubierta para cada personas. Después convertiremos el número restante en una variable numérica. Los valores que faltan se convertirán en cero. En la imagen de abajo se pueden ver las cubiertas reales del titánico, que van de la A a la G.

# In[36]:



deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data = [train_df, test_df]

for dataset in data:
        dataset ['Cabin'] = dataset['Cabin'].fillna("U0")
        dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        dataset['Deck'] = dataset['Deck'].map(deck)
        dataset['Deck'] = dataset['Deck'].fillna(0)
        dataset['Deck'] = dataset['Deck'].astype(int)
# we can now drop the cabin feature
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)


# In[35]:


train_df.head()


# La edad:
# Ahora podemos abordar el tema con las características de la edad que faltan en los valores. Crearé una matriz que contiene números aleatorios, que se calculan en base al valor medio de la edad con respecto a la desviación estándar y es_nula.

# In[ ]:


data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)
train_df["Age"].isnull().sum()


# Embarcaque:
# Dado que el rasgo Embarcado tiene sólo 2 valores faltantes, sólo los llenaremos con el más común.

# In[ ]:


train_df['Embarked'].describe()


# In[ ]:


common_value = 'S'
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)


# In[ ]:


train_df.info()


# In[ ]:


#transformando caracteristicas
train_df.info()


# In[ ]:


train_df.head()


# Arriba pueden ver que 'tarifa' es un float y tenemos que lidiar con 4 características categóricas: Nombre, Sexo, Billete y embarque. 
# 
# 

# Tarifa:
# Convirtiendo "Tarifa" de float a int64, usando la función "astype()" que proporciona pandas:

# In[ ]:


data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)


# Nombre:
# Usaremos la característica del Nombre para extraer los Títulos del Nombre, para poder construir una nueva característica a partir de eso.

# In[ ]:


data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)


# Sexo:
# Convierte  "Sexo" en numérica.

# In[ ]:


genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)


# ticket:
# Dado que el atributo Ticket tiene 681 tickets únicos, será un poco difícil convertirlos en categorías útiles. Así que lo eliminaremos del conjunto de datos.

# In[ ]:


train_df['Ticket'].describe()


# In[ ]:


train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)


# Embarque:
# Convierte la característica "Embarcado" en numérica.

# In[ ]:


ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)


# In[ ]:


#Creando categorías:
#edad
data = [train_df, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
    
print(train_df['Age'].value_counts())


# In[ ]:


#tarifa
#podemos usar la función sklearn "qcut()", que podemos usar para ver, cómo podemos formar las categorías.
data = [train_df, test_df]


for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[ ]:


#creando nuevas variables

#clases de edad
data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']


# In[ ]:


#tarifa por persona
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
# Let's take a last look at the training set, before we start training the models.
train_df.head(10)


# In[ ]:


#Machine Learning
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()


# In[ ]:


#Random Forest:
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)


# In[ ]:


#Logistic Regression:
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)


# In[ ]:


#Perceptron:
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)


# In[ ]:


X_test# Linear Support Vector Machine:
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)


# In[ ]:


#Decision Tree
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, Y_train)  
Y_pred = decision_tree.predict(X_test)  
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)


# In[ ]:


results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Logistic Regression', 
              'Random Forest', 'Perceptron', 
              'Decision Tree'],
    'Score': [acc_linear_svc,  acc_log, 
              acc_random_forest,  acc_perceptron, 
               acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




