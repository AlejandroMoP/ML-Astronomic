
# coding: utf-8

# In[39]:


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
means = features[['Mean_1', 'Mean_2']].copy()
means.dropna(inplace=True)

plt.subplot(1, 2, 2)
plt.scatter(
    means.values[:, 0] - means.values[:, 1],
    means.values[:, 0],
    alpha=0.005
)
plt.ylabel('Mean_1')
plt.xlabel('Mean_1 - Mean_2')
plt.title('g mean magnitude vs "color" for ALeRCE - ZTF dataset')
plt.show()


# ### Creación del nuevo dataframe

# In[6]:


DFclas=features.loc[labels.index]
DFclas=DFclas.dropna()

#para visualizar las columnas y filas de los dataframes usar los siguientes:
#DFclas.head()
#labels.dropna()
#labels.head()


# In[8]:


a=[]
for i in range(len(DFclas.index)):
    a.append(labels.loc[DFclas.index[i]][0])

#Se añade una nueva columna con la clase   
DFclas['clase']=a
DFclas.head()


# In[9]:


#Visualizacion de ejemplos por clase para el nuevo dataframe
DFclas.groupby('clase').count()


# In[11]:


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

SNIa=DFclas['clase']=='SNIa'  #menos de 500 
dfSNIa=DFclas[SNIa]


DF5=pd.concat([dfRRL[0:500],dfAGNI[0:500],dfBlazar[0:500],dfCVNova[0:500],dfEBC[0:500],dfEBSDD[0:500],dfLPV[0:500],dfPO[0:500]])

DF1=pd.concat([dfAGNI[0:100],dfBlazar[0:100],dfCVNova[0:100],dfEBC[0:100],dfEBSDD[0:100],dfLPV[0:100],dfPO[0:100],dfRRL[0:100],
               dfCeph[0:100],dfDSCT[0:100],dfSNIa[0:100]])

#Si es que se desea visualizar usar los siguientes:
#DF5.groupby('clase').count()
#DF1.groupby('clase').count()


# ### Comparación de caracteristicas

# In[17]:


#Para 500 ejemplos

y5 = DF5.iloc[:,96].values
means5 = DF5[['Mean_1', 'Mean_2']].copy()
#DF5.dropna(inplace=True)
plt.figure(figsize=(8, 6))
for lab, col in zip(('RRL', 'EBC', 'EBSD/D','AGN-I','CV/Nova','Blazar','LPV','Periodic-Other'),
                        ('magenta', 'cyan', 'limegreen','blue','r','y','brown','g')):
    plt.scatter(
                means5.values[y5==lab, 0] - means5.values[y5==lab, 1],
                means5.values[y5==lab, 0],label=lab,c=col, alpha=0.5)
plt.ylabel('Mean_1')
plt.xlabel('Mean_1 - Mean_2')
plt.legend(loc='lower left')
plt.title('g mean magnitude vs "color" for ALeRCE - ZTF dataset')


# In[22]:


#Para 100 ejemplos por clase

y1 = DF1.iloc[:,96].values
means1 = DF1[['Mean_1', 'Mean_2']].copy()
#DF5.dropna(inplace=True)
plt.figure(figsize=(8, 6))
for lab, col in zip(('RRL', 'EBC', 'EBSD/D','AGN-I','CV/Nova','Blazar','LPV','Periodic-Other','Ceph','DSCT','SNIa'),
                        ('magenta', 'cyan', 'limegreen','blue','r','y','brown','g','indigo','orange','gray')):
    plt.scatter(
                means1.values[y1==lab, 0] - means1.values[y1==lab, 1],
                means1.values[y1==lab, 0],label=lab,c=col, alpha=0.5)
plt.ylabel('Mean_1')
plt.xlabel('Mean_1 - Mean_2')
plt.legend(loc='lower left')
plt.title('g mean magnitude vs "color" for ALeRCE - ZTF dataset')


# In[24]:


y5 = DF5.iloc[:,96].values
amplitud5 = DF5[['Amplitude_1', 'Amplitude_2']].copy()
plt.figure(figsize=(8, 6))
for lab, col in zip(('RRL', 'EBC', 'EBSD/D','AGN-I','CV/Nova','Blazar','LPV','Periodic-Other'),
                        ('magenta', 'cyan', 'limegreen','blue','r','y','brown','g')):
    plt.scatter(
                amplitud5.values[y5==lab, 0] - amplitud5.values[y5==lab, 1],
                amplitud5.values[y5==lab, 0],label=lab,c=col, alpha=0.5)
plt.ylabel('amplitude_1')
plt.xlabel('amplitude_1 - amplitude_2')
plt.legend(loc='lower left')
plt.title('Amplitud 1 vs diferencia de amplitudes  ALeRCE - ZTF dataset')


# In[25]:


y1 = DF1.iloc[:,96].values
amplitud1 = DF1[['Amplitude_1', 'Amplitude_2']].copy()
plt.figure(figsize=(8, 6))
for lab, col in zip(('RRL', 'EBC', 'EBSD/D','AGN-I','CV/Nova','Blazar','LPV','Periodic-Other','Ceph','DSCT','SNIa'),
                        ('magenta', 'cyan', 'limegreen','blue','r','y','brown','g','indigo','orange','gray')):
    plt.scatter(
                amplitud1.values[y1==lab, 0] - amplitud1.values[y1==lab, 1],
                amplitud1.values[y1==lab, 0],label=lab,c=col, alpha=0.5)
plt.ylabel('amplitude_1')
plt.xlabel('amplitude_1 - amplitude_2')
plt.legend(loc='lower left')
plt.title('Amplitud 1 vs diferencia de amplitudes  ALeRCE - ZTF dataset')


# ## Analisis de componentes principales(PCA)

# ### Para 500 ejemplos

# In[29]:


# Se desea ver como se relacionan las componentes principales de la base de datos.
# En los tres graficos siguientes se ve la relación de las 3 componentes principales, una versus otra.

Xeq = DF5.iloc[:,0:96].values
#X = l.iloc[:,:3].values
yeq = DF5.iloc[:,96].values
#y = l.iloc[:,0].values
Xeq_std = StandardScaler().fit_transform(Xeq)
cov_mat2eq = np.cov(Xeq_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat2eq)

#visualizar los vecotores y valores propios, aqui se encuentran los componentes principales
#print('Eigenvectors \n%s' %eig_vecs)
#print('\nEigenvalues \n%s' %eig_vals)
#  Hacemos una lista de parejas (autovector, autovalor) 
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Ordenamos estas parejas en orden descendiente con la función sort
eig_pairs.sort(key=lambda x: x[0], reverse=True)
matrix_w = np.hstack((eig_pairs[0][1].reshape(96,1),
                      eig_pairs[1][1].reshape(96,1)))


#print('Matriz W:\n', matrix_w) 

Yeq = Xeq_std.dot(matrix_w)
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(8, 6))
    for lab, col in zip(('RRL','AGN-I','Blazar','CV/Nova','EBC','EBSD/D','LPV','Periodic-Other'),
                        #('RRL', 'EBC', 'EBSD/D','AGN-I','Blazar','CV/Nova')
                        ('magenta', 'cyan', 'limegreen','blue','r','y','brown','g')):
        plt.scatter(Yeq[yeq==lab, 0],
                    Yeq[yeq==lab, 1],
                    label=lab,
                    c=col, alpha=0.55)
    plt.xlabel('Comp Principal 1 N-Samples')
    plt.ylabel('Comp Principal 2 Amplitude-1')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


# In[33]:


matrix_w = np.hstack((eig_pairs[1][1].reshape(96,1),
                      eig_pairs[2][1].reshape(96,1)))

#print('Matriz W:\n', matrix_w)

Yeq = Xeq_std.dot(matrix_w)

plt.figure(figsize=(8, 6))
for lab, col in zip(('RRL','AGN-I','Blazar','CV/Nova','EBC','EBSD/D','LPV','Periodic-Other'),
                        #('RRL', 'EBC', 'EBSD/D','AGN-I','Blazar','CV/Nova')
                        ('magenta', 'cyan', 'limegreen','blue','r','y','brown','g')):
        plt.scatter(Yeq[yeq==lab, 0],
                    Yeq[yeq==lab, 1],
                    label=lab,
                    c=col,alpha=0.55)
plt.xlabel('Comp Principal 2 Amplitude-1')
plt.ylabel('Comp Principal 3 AndersonDarling-1')
plt.title('Aplicacion PCA')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


# In[34]:


matrix_w = np.hstack((eig_pairs[0][1].reshape(96,1),
                      eig_pairs[2][1].reshape(96,1)))

#print('Matriz W:\n', matrix_w)

Yeq = Xeq_std.dot(matrix_w)

plt.figure(figsize=(8, 6))
for lab, col in zip(('RRL','AGN-I','Blazar','CV/Nova','EBC','EBSD/D','LPV','Periodic-Other'),
                        #('RRL', 'EBC', 'EBSD/D','AGN-I','Blazar','CV/Nova')
                        ('magenta', 'cyan', 'limegreen','blue','r','y','brown','g')):
        plt.scatter(Yeq[yeq==lab, 0],
                    Yeq[yeq==lab, 1],
                    label=lab,
                    c=col,alpha=0.55)
plt.xlabel('Comp Principal 1 N-Samples')
plt.ylabel('Comp Principal 3 AndersonDarling-1')
plt.title('Aplicacion PCA')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


# ### Para 100 ejemplos

# In[35]:


Xeq = DF1.iloc[:,0:96].values
#X = l.iloc[:,:3].values
yeq = DF1.iloc[:,96].values
#y = l.iloc[:,0].values
Xeq_std = StandardScaler().fit_transform(Xeq)
cov_mat2eq = np.cov(Xeq_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat2eq)

#print('Eigenvectors \n%s' %eig_vecs)
#print('\nEigenvalues \n%s' %eig_vals)
#  Hacemos una lista de parejas (autovector, autovalor) 
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Ordenamos estas parejas den orden descendiente con la función sort
eig_pairs.sort(key=lambda x: x[0], reverse=True)
matrix_w = np.hstack((eig_pairs[0][1].reshape(96,1),
                      eig_pairs[1][1].reshape(96,1)))

#print('Matriz W:\n', matrix_w)

Yeq = Xeq_std.dot(matrix_w)
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(8, 6))
    for lab, col in zip(('RRL','AGN-I','Blazar','CV/Nova','EBC','EBSD/D','LPV','Periodic-Other','Ceph','DSCT','SNIa'),
                        #('RRL', 'EBC', 'EBSD/D','AGN-I','Blazar','CV/Nova')
                        ('magenta', 'cyan', 'limegreen','blue','r','y','brown','g','indigo','orange','gray')):
        plt.scatter(Yeq[yeq==lab, 0],
                    Yeq[yeq==lab, 1],
                    label=lab,
                    c=col,alpha=0.55)
    plt.xlabel('Comp Principal 1 N-samples')
    plt.ylabel('Comp Principal 2 Amplitude-1')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


# In[36]:


matrix_w = np.hstack((eig_pairs[1][1].reshape(96,1),
                      eig_pairs[2][1].reshape(96,1)))

#print('Matriz W:\n', matrix_w)

Yeq = Xeq_std.dot(matrix_w)

plt.figure(figsize=(8, 6))
for lab, col in zip(('RRL','AGN-I','Blazar','CV/Nova','EBC','EBSD/D','LPV','Periodic-Other','Ceph','DSCT','SNIa'),
                        #('RRL', 'EBC', 'EBSD/D','AGN-I','Blazar','CV/Nova')
                        ('magenta', 'cyan', 'limegreen','blue','r','y','brown','g','indigo','orange','gray')):
        plt.scatter(Yeq[yeq==lab, 0],
                    Yeq[yeq==lab, 1],
                    label=lab,
                    c=col)
plt.xlabel('Comp Principal 2 Amplitude-1')
plt.ylabel('Comp Principal 3 AndersonDarling-1')
plt.title('Aplicacion PCA')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# In[37]:


matrix_w = np.hstack((eig_pairs[0][1].reshape(96,1),
                      eig_pairs[2][1].reshape(96,1)))

#print('Matriz W:\n', matrix_w)

Yeq = Xeq_std.dot(matrix_w)

plt.figure(figsize=(8, 6))
for lab, col in zip(('RRL','AGN-I','Blazar','CV/Nova','EBC','EBSD/D','LPV','Periodic-Other','Ceph','DSCT','SNIa'),
                        #('RRL', 'EBC', 'EBSD/D','AGN-I','Blazar','CV/Nova')
                        ('magenta', 'cyan', 'limegreen','blue','r','y','brown','g','indigo','orange','gray')):
        plt.scatter(Yeq[yeq==lab, 0],
                    Yeq[yeq==lab, 1],
                    label=lab,
                    c=col)
plt.xlabel('Comp Principal 1 N-samples')
plt.ylabel('Comp Principal 3 AndersonDarling-1')
plt.title('Aplicacion PCA')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# ## Aplicación de T-sne

# ### para 100 ejemplos

# In[43]:


tsne = TSNE(n_components=2, random_state=0)
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
                       'dodgerblue','green','lightgreen','gray','aqua','pink',],
              legend='full',
              data=tsne_df);
plt.title('t-SNE')


# ### T-sne usando las 20 componentes mas representativas

# In[48]:


# Debido a que no todas las caracteristicas influyen de la misma manera se usaran las 20 más influyentes
#para 100 ejemplos

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000,random_state=0)
y=DF1.iloc[:,96]
#data_X = digits.data[:600]
tsne_obj= tsne.fit_transform(DF1.iloc[:,0:20])
tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'clases':y})

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(x="X", y="Y",
              hue="clases",
              palette=['magenta', 'cyan', 'limegreen','blue','r','y','brown','g','indigo','orange','gray'],
              legend='full',
              data=tsne_df);
plt.title('t-SNE con Periodic-Other')


# In[49]:

#Para 500 ejemplos

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000,random_state=0)
y=DF5.iloc[:,96]
#data_X = digits.data[:600]
tsne_obj= tsne.fit_transform(DF5.iloc[:,0:20])
tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'clases':y})

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(x="X", y="Y",
              hue="clases",
              palette=['magenta', 'cyan', 'limegreen','blue','r','y','brown','g'],
              legend='full',
              data=tsne_df);
plt.title('t-SNE con Periodic-Other')

