#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
            discriminant_analysis, random_projection)


# In[2]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


detections = pd.read_pickle('C:/Users/sebas/Desktop/Ingenieria/8 semestre/Inteligencia Computacional/proyecto/data_representativity/detections_664k.pkl')
features = pd.read_pickle('C:/Users/sebas/Desktop/Ingenieria/8 semestre/Inteligencia Computacional/proyecto/data_representativity/features_664k.pkl')
labels = pd.read_pickle('C:/Users/sebas/Desktop/Ingenieria/8 semestre/Inteligencia Computacional/proyecto/data_representativity/labels.pkl')

print(detections.head())
print(labels.head())

# How are the classes distributed?
print(labels[['classALeRCE', 'ra']].groupby('classALeRCE').count())

# Plot an object
first_object_oid = detections.index.values[0]#20760278
first_object_detections = detections.loc[first_object_oid]

print(f'Object {first_object_oid} has {len(first_object_detections)} detections')

plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(
    first_object_detections.mjd,
    first_object_detections.magpsf_corr,
    c=first_object_detections.fid)

# In astronomy we plot the magnitude axis inverted (higher magnitude, dimmer objects)
plt.gca().invert_yaxis()
plt.xlabel('Time [mjd]')
plt.ylabel('Magnitude')
plt.title(f'{first_object_oid} light curve')

# Let's do a simple scatter of two features
#means = features[['Mean_1', 'Mean_2']].copy()
#means.dropna(inplace=True)

#plt.subplot(1, 2, 2)
#plt.scatter(
   # means.values[:, 0] - means.values[:, 1],
    #means.values[:, 0],
    #alpha=0.005
#)
#plt.ylabel('Mean_1')
#plt.xlabel('Mean_1 - Mean_2')
#plt.title('g mean magnitude vs "color" for ALeRCE - ZTF dataset')
#plt.show()


# ### Creación del nuevo dataframe

# In[3]:


DFclas=features.loc[labels.index]
DFclas=DFclas.dropna()

#para visualizar las columnas y filas de los dataframes usar los siguientes:
#DFclas.head()
#labels.dropna()
#labels.head()


# In[4]:


a=[]
for i in range(len(DFclas.index)):
    a.append(labels.loc[DFclas.index[i]][0])

#Se añade una nueva columna con la clase   
DFclas['clase']=a
#DFclas.head()


# In[5]:


#Visualizacion de ejemplos por clase para el nuevo dataframe
DFclas.groupby('clase').count()


# ### Conjunto de supernovas

# In[6]:


S1=DFclas['clase']=='SLSN'
S2=DFclas['clase']=='SNII'
S3=DFclas['clase']=='SNIIb'
S4=DFclas['clase']=='SNIIn'
S5=DFclas['clase']=='SNIa'
S6=DFclas['clase']=='SNIbc'

#Hay que añadir CV/nova??

dfS1=DFclas[S1]
dfS2=DFclas[S2]
dfS3=DFclas[S3]
dfS4=DFclas[S4]
dfS5=DFclas[S5]
dfS6=DFclas[S6]

SN=pd.concat([dfS1,dfS2,dfS3,dfS4,dfS5,dfS6])
SN['clase']='SN'
#SN


# In[7]:


import random 
n=500
aleatorios = [random.randint(1,512) for _ in range(n)]

SN=SN.dropna()
dfSN=SN.iloc[aleatorios]
#dfSN


# In[8]:


# Se creara el dataframe que contenga 100 ejemplos de cada clase y otro con 500 ejemplos
# Las clases con menos ejemplos que los indicados no se tomaran en cuenta.

AGNI=DFclas['clase']=='AGN-I'
dfAGNI=DFclas[AGNI]

Blazar=DFclas['clase']=='Blazar'
dfBlazar=DFclas[Blazar]

CVNova=DFclas['clase']=='CV/Nova'
dfCVNova=DFclas[CVNova]

Ceph=DFclas['clase']=='Ceph' #menos de 500
dfCeph=DFclas[Ceph]

DSCT=DFclas['clase']=='DSCT' #menos de 500
dfDSCT=DFclas[DSCT]

EBC=DFclas['clase']=='EBC'
dfEBC=DFclas[EBC]

EBSDD=DFclas['clase']=='EBSD/D'
dfEBSDD=DFclas[EBSDD]

LPV=DFclas['clase']=='LPV'
dfLPV=DFclas[LPV]

PeriodicOther=DFclas['clase']=='Periodic-Other'
dfPO=DFclas[PeriodicOther]

RRL1000=DFclas['clase']=='RRL'
dfRRL=DFclas[RRL1000] 

#SNIa=DFclas['clase']=='SNIa'  #menos de 500 
#dfSNIa=DFclas[SNIa]


# Estos son los dos conjuntos finales


DF5=pd.concat([dfRRL[0:500],dfAGNI[0:500],dfBlazar[0:500],dfCVNova[0:500],dfEBC[0:500],dfEBSDD[0:500],dfLPV[0:500],dfPO[0:500],dfSN[0:500]])

DF1=pd.concat([dfAGNI[0:100],dfBlazar[0:100],dfCVNova[0:100],dfEBC[0:100],dfEBSDD[0:100],dfLPV[0:100],dfPO[0:100],dfRRL[0:100],
               dfCeph[0:100],dfDSCT[0:100],dfSN[0:100]])

#Si es que se desea visualizar usar los siguientes:
#DF5.groupby('clase').count()
#DF1.groupby('clase').count()


# ## Comparación de caracteristicas

# ### Para 100 ejemplos

# # Analisis t-SNE

# ### Utilizando la base de datos equilibrada

# In[9]:


tsne = TSNE(n_components=2, random_state=0, perplexity=40)
y=DF1.iloc[:,96]
#data_X = digits.data[:600]
tsne_obj= tsne.fit_transform(DF1.iloc[:,0:95])
tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'clases':y})

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(x="X", y="Y",
              hue="clases",
              palette=['purple','red','orange','brown','blue',
                       'dodgerblue','green','lightgreen','black','aqua','pink',],
              legend='full',
              data=tsne_df);
plt.title('t-SNE')


# ## Encontrar las caracteristicas mas representativas
# 
# Se usara random forest para eliminar las caracteristicas que tengan poca relevancia, para esto se espera tener una importancia de las caracteristicas del 80% y 90%.

# In[11]:


#muestra el conjunto de datos sin los target
#X = DF5.drop(['clase'],axis='columns')
X = DFalldata.drop(['clase'],axis='columns')
#muestra el vector de los target de DFclas
#Y=DF5.iloc[:,-1]
Y=DFalldata.iloc[:,-1]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.5, stratify=Y)

classifier = RandomForestClassifier(n_estimators=40,max_depth=30)
classifier.fit(X_train, Y_train)

feature_importances = classifier.feature_importances_
importance_order = np.argsort(-feature_importances)
feature_names = DFalldata.columns.values


# In[20]:


#Numero de caracteristicas para llegar al 80%

suma=0
indice=0
algo=np.sort(feature_importances)[::-1]
for i in range(len(algo)):
    if suma<=0.8:
        suma = suma+algo[i]
        indice=i
print(indice)
print(suma)


# In[21]:



print('\nCaracteristicas ordenadas por importancia (RF)')
for index in importance_order:
    print('\t%.3f %s' % (feature_importances[index], feature_names[index]))


# ## Agregar elementos no etiquetados

# In[10]:


import random 
n=1199
aleatorios = [random.randint(0,256983) for _ in range(n)]

dfsinetiq = features.drop(labels.index)
dfsinetiq=dfsinetiq.dropna()
df=dfsinetiq.iloc[aleatorios]
df['clase']='sin-clase'
DFalldata=pd.concat([DF1,df])  # Toda la data

dff=DFalldata.iloc[:,0:96]
data_norm=(dff-dff.min())/(dff.max()-dff.min())
data_norm2=data_norm
data_norm2['clase']=DFalldata['clase']


# In[13]:


#NuevaData = DFalldata[DFalldata.columns.values[importance_order[0:53]]]
NuevaData1 = DFalldata[DFalldata.columns.values[importance_order[0:59]]]
dff=NuevaData1
tsne = TSNE(n_components=2, random_state=0, perplexity=70)
y=DFalldata.iloc[:,96]
#data_X = digits.data[:600]
data_norm=(dff-dff.min())/(dff.max()-dff.min())
tsne_obj= tsne.fit_transform(data_norm)
tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'clases':y})

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(x="X", y="Y",
              hue="clases",
              palette=['purple','red','orange','brown','blue',
                       'dodgerblue','green','lightgreen','black','aqua','pink','darkkhaki',],
              legend='full',
              data=tsne_df,alpha=0.7);
plt.title('t-SNE')

#El grafico obtenido difiere del mostrado en el informe debido a que el t SNE se corrio de nuevo, pero el analisis es el mismo.


# ## Reconocer cluster de elementos sin clase

# ### Curva de luz de un elemento del cluster 1

# In[14]:


#Para el cluster de la izq
#Las ubicaciones dependen de como resulto el t-SNE y cuales elementos sin clase salieron del random


arreglosx=[]
arreglosy=[]
for i in range(len(tsne_obj)):
    if tsne_obj[i][0]<10 and tsne_obj[i][0]>0 and tsne_obj[i][1]<-15 and tsne_obj[i][1]>-30:
        arreglosx.append(tsne_obj[i][0])
        arreglosy.append(tsne_obj[i][1])
        
nelem=[]
for j in range(len(arreglosx)):
    for i in range(len(tsne_obj)):
        if tsne_obj[i][0]== arreglosx[j]:
            nelem.append(i)

        
sinclas=[]
for k in range(len(nelem)):
    if nelem[k]>1198:
        sinclas.append(nelem[k])
        
Cjto1=dff.index[sinclas] 
Cjto1


# In[3]:


#curvas de luz de los elementos obtenidos en el primer cluster

object_oid='ZTF18aajswmq'
#object_oid='ZTF18aazmvhd'
object_detections = detections.loc[object_oid]
object_features = features.loc[object_oid]

mjd = object_detections.mjd #tabla
periodo = float(object_features.PeriodLS_v2_1) #numero fijo

len(mjd)
fase = []
for i in mjd:
    fase.append((i%periodo)/periodo)

#Curva de Luz doblada    
plt.figure(figsize=(15,4))
plt.subplot()
plt.scatter(
    fase,
    object_detections.magpsf_corr,
    c=object_detections.fid)
plt.gca().invert_yaxis()
plt.xlabel('Phase')
plt.ylabel('Magnitude')
plt.title(f'{object_oid} light curve')


#Curva de luz no doblada
plt.figure(figsize=(15,4))
plt.subplot()
plt.scatter(
    object_detections.mjd,
    object_detections.magpsf_corr,
    c=object_detections.fid)

plt.gca().invert_yaxis()
plt.xlabel('Time [mjd]')
plt.ylabel('Magnitude')
plt.title(f'{object_oid} light curve')


# ### Curva de luz del cluster 2

# In[15]:


#Para el cluster de abajo a la der

arreglosx=[]
arreglosy=[]
for i in range(len(tsne_obj)):
    if tsne_obj[i][0]<24 and tsne_obj[i][0]>17 and tsne_obj[i][1]<-17 and tsne_obj[i][1]>-27:
        arreglosx.append(tsne_obj[i][0])
        arreglosy.append(tsne_obj[i][1])
        
nelem=[]
for j in range(len(arreglosx)):
    for i in range(len(tsne_obj)):
        if tsne_obj[i][0]== arreglosx[j]:
            nelem.append(i)

        
sinclas=[]
for k in range(len(nelem)):
    if nelem[k]>1198:
        sinclas.append(nelem[k])
        
Cjto1=dff.index[sinclas] 
Cjto1


# In[4]:


#curvas de luz de los elementos obtenidos en el segundo cluster

object_oid='ZTF18abokskf'
#object_oid='ZTF18abmfwyv'
object_detections = detections.loc[object_oid]
object_features = features.loc[object_oid]

mjd = object_detections.mjd #tabla
periodo = float(object_features.PeriodLS_v2_1) #numero fijo

len(mjd)
fase = []
for i in mjd:
    fase.append((i%periodo)/periodo)

#Curva de Luz doblada    
plt.figure(figsize=(15,4))
plt.subplot()
plt.scatter(
    fase,
    object_detections.magpsf_corr,
    c=object_detections.fid)
plt.gca().invert_yaxis()
plt.xlabel('Phase')
plt.ylabel('Magnitude')
plt.title(f'{object_oid} light curve')


#Curva de luz no doblada
plt.figure(figsize=(15,4))
plt.subplot()
plt.scatter(
    object_detections.mjd,
    object_detections.magpsf_corr,
    c=object_detections.fid)

plt.gca().invert_yaxis()
plt.xlabel('Time [mjd]')
plt.ylabel('Magnitude')
plt.title(f'{object_oid} light curve')


# ### Curvas de luz del cluster 3

# In[16]:


#Para el cluster de centro izquierda
arreglosx=[]
arreglosy=[]
for i in range(len(tsne_obj)):
    if tsne_obj[i][0]<31 and tsne_obj[i][0]>17 and tsne_obj[i][1]<-7 and tsne_obj[i][1]>-15:
        arreglosx.append(tsne_obj[i][0])
        arreglosy.append(tsne_obj[i][1])
        
nelem=[]
for j in range(len(arreglosx)):
    for i in range(len(tsne_obj)):
        if tsne_obj[i][0]== arreglosx[j]:
            nelem.append(i)

        
sinclas=[]
for k in range(len(nelem)):
    if nelem[k]>1198:
        sinclas.append(nelem[k])
        
Cjto1=dff.index[sinclas] 
Cjto1


# In[5]:


#curvas de luz de los elementos obtenidos en el tercer cluster

object_oid='ZTF18abbsnps'
object_detections = detections.loc[object_oid]
object_features = features.loc[object_oid]

mjd = object_detections.mjd #tabla
periodo = float(object_features.PeriodLS_v2_1) #numero fijo

len(mjd)
fase = []
for i in mjd:
    fase.append((i%periodo)/periodo)

#Curva de Luz doblada    
plt.figure(figsize=(15,4))
plt.subplot()
plt.scatter(
    fase,
    object_detections.magpsf_corr,
    c=object_detections.fid)
plt.gca().invert_yaxis()
plt.xlabel('Phase')
plt.ylabel('Magnitude')
plt.title(f'{object_oid} light curve')


#Curva de luz no doblada
plt.figure(figsize=(15,4))
plt.subplot()
plt.scatter(
    object_detections.mjd,
    object_detections.magpsf_corr,
    c=object_detections.fid)

plt.gca().invert_yaxis()
plt.xlabel('Time [mjd]')
plt.ylabel('Magnitude')
plt.title(f'{object_oid} light curve')


# # Aplicacion de Kmeans

# In[33]:


from mpl_toolkits.mplot3d import Axes3D 

from sklearn.cluster import KMeans 
from matplotlib import colors as mcolors 
import math 


# In[34]:


clusters = 15
  
model = KMeans(n_clusters = clusters) 
model.fit(data_norm2.iloc[:,0:96]) 

md_k = pd.Series(model.labels_)


# In[35]:


data_norm2['Clust'] = model.labels_
data_norm2.tail()


# In[36]:


tres=data_norm2['Clust']==0
data3=data_norm2[tres]
v3=data3['clase']=='sin-clase'

print('La cantidad de elementos sin-clase en este cluster es de '+ str(len(data3[v3]))+', con un total de '+str(len(v3))+' elementos' )

tres=data_norm2['Clust']==1
data3=data_norm2[tres]
v3=data3['clase']=='sin-clase'

print('La cantidad de elementos sin-clase en este cluster es de '+ str(len(data3[v3]))+', con un total de '+str(len(v3))+' elementos' )


# In[37]:


for i in range(0,15):
    
    tres=data_norm2['Clust']==i
    data3=data_norm2[tres]
    v3=data3['clase']=='sin-clase'

    print('La cantidad de elementos sin-clase en este cluster ' + str(i) +' es de '+ str(len(data3[v3]))+', con un total de '+str(len(v3))+
          ' elementos, con un ' + str("{0:.4f}".format(len(data3[v3])*100/len(v3)))+'%' )


# In[38]:


NuevaData1 = DFalldata[DFalldata.columns.values[importance_order[0:59]]]
dff=NuevaData1
y=data_norm2.iloc[:,97]
#y=data_norm.iloc[:,95]
#data_norm2=(dff-dff.min())/(dff.max()-dff.min())
tsne_obj= tsne.fit_transform(data_norm2.iloc[:,0:96])
tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'clases':y})

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(x="X", y="Y",
              hue="clases",
              palette=['purple','red','orange','brown','blue',
                       'dodgerblue','green','lightgreen','gray','black','aqua','pink','darkkhaki','salmon','fuchsia',],#'crimson',],
              legend='full',
              data=tsne_df);
plt.title('t-SNE')

