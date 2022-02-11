#!/usr/bin/env python
# coding: utf-8

# # clase 9

# In[2]:


import seaborn as sns

import pandas as pd
company = pd.read_csv('company_sales_data.csv')
print(company.head())


# In[3]:


company


# In[4]:


import matplotlib.pyplot as plt


# Ejercicio 1: lee el beneficio total de todos los meses y muestralos usando un gráfico a tu elección

# In[5]:


company.groupby("month_number").total_profit.mean()[:].plot()


# Ejercicio 2: lee todos los datos de ventas de productos y muestralos usando  un gráfico a tu elección

# In[6]:


company=company.drop(['total_units','total_profit'],axis=1)
company.plot.area()


# Eliminamos las filas que no queremos mostrar y asi solo observamos los productos con sus ventas

# In[ ]:





# In[ ]:




