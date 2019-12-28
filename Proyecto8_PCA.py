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
import random
from sklearn.preprocessing import normalize


# In[2]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


detections = pd.read_pickle(r'C:\Users\aleja\Desktop\p8ia\data_representativity\detections_664k.pkl')
features = pd.read_pickle(r'C:\Users\aleja\Desktop\p8ia\data_representativity\features_664k.pkl')
labels = pd.read_pickle(r'C:\Users\aleja\Desktop\p8ia\data_representativity\labels.pkl')

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


# In[3]:


plt.scatter(
    first_object_detections.mjd,
    first_object_detections.magpsf_corr,
    c=first_object_detections.fid)

# In astronomy we plot the magnitude axis inverted (higher magnitude, dimmer objects)
plt.gca().invert_yaxis()
plt.xlabel('Time [mjd]')
plt.ylabel('Magnitude')
plt.title(f'{first_object_oid} light curve')


# In[4]:


means = features[['Mean_1', 'Mean_2']].copy()
means.dropna(inplace=True)
plt.scatter(
    means.values[:, 0] - means.values[:, 1],
    means.values[:, 0],
    alpha=0.3)
plt.ylabel('Mean_1')
plt.xlabel('Mean_1 - Mean_2')
plt.title('g mean magnitude vs "color" for ALeRCE - ZTF dataset')


# In[5]:


features.head()
len(features)
indices=features.index
indices.sort


# In[6]:


#detections.drop_duplicates(subset =["fid"] , 
                     #keep = False, inplace = True) 
labels.head()

l=labels.iloc[:,0:3]
l=l.dropna()


# In[7]:


y = l.iloc[:,0].values
y


# In[8]:



p=features.iloc[:,0:96]
p=p.dropna()


# In[9]:



X = p.iloc[:,0:96].values
#X = l.iloc[:,:3].values
y = p.iloc[:,0].values
#y = l.iloc[:,0].values

X_std = StandardScaler().fit_transform(X)
cov_mat = np.cov(X_std.T)


# In[10]:



cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

#print('Eigenvectors \n%s' %eig_vecs)
#print('\nEigenvalues \n%s' %eig_vals)


# In[11]:



#  Hacemos una lista de parejas (autovector, autovalor) 
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Ordenamos estas parejas den orden descendiente con la función sort
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visualizamos la lista de autovalores en orden desdenciente

#for i in eig_pairs:
 #   print(i[0])


# In[12]:



#Generamos la matríz a partir de los pares autovalor-autovector
matrix_w = np.hstack((eig_pairs[0][1].reshape(96,1),
                      eig_pairs[1][1].reshape(96,1)))



Y = X_std.dot(matrix_w)


# In[13]:


DFclas=features.loc[labels.index]


# In[14]:


DFclas=DFclas.dropna()


# In[15]:


a=[]
for i in range(len(DFclas.index)):
    a.append(labels.loc[DFclas.index[i]][0])
    


# In[17]:


DFclas2=DFclas


# In[18]:


DFclas['clase']=a


# In[21]:


RRL1000=DFclas['clase']=='RRL'
dfRRL=DFclas[RRL1000]
AGNI=DFclas['clase']=='AGN-I'
dfAGNI=DFclas[AGNI]
Blazar=DFclas['clase']=='Blazar'
dfBlazar=DFclas[Blazar]
CVNova=DFclas['clase']=='CV/Nova'
dfCVNova=DFclas[CVNova]
EBC=DFclas['clase']=='EBC'
dfEBC=DFclas[EBC]
EBSDD=DFclas['clase']=='EBSD/D'
dfEBSDD=DFclas[EBSDD]
LPV=DFclas['clase']=='LPV'
dfLPV=DFclas[LPV]
PeriodicOther=DFclas['clase']=='Periodic-Other'
dfPO=DFclas[PeriodicOther]
SNIa=DFclas['clase']=='SNIa'
dfSNIa=DFclas[SNIa]
Ceph=DFclas['clase']=='Ceph'
dfCeph=DFclas[Ceph]
DSCT=DFclas['clase']=='DSCT'
dfDSCT=DFclas[DSCT]
SNII=DFclas['clase']=='SNII'
dfSNII=DFclas[SNII]
SNIa=DFclas['clase']=='SNIa'
dfSNIa=DFclas[SNIa]


# In[ ]:


AGN-I            8798si
AGN-II             56no
Blazar            755si
CV/Nova           708si
Ceph              563no 360 habian
DSCT              663no 490 habian
EBC              6554si
EBSD/D          24442si
LPV              2856si
Periodic-Other    952si
RRL             15176si
SLSN               14no
SNII              142no
SNIIb               9no
SNIIn              22no
SNIa              569no 357
SNIbc              39no
TDE                 3no
ZZ                  2no


# In[23]:


DFequal2=pd.concat([dfRRL[0:100],dfAGNI[0:100],dfBlazar[0:100],dfCVNova[0:100],dfEBC[0:100],dfEBSDD[0:100],dfLPV[0:100],dfPO[0:100],dfCeph[0:100],dfDSCT[0:100],dfSNII[0:100],dfSNIa[0:100]])
DFequal2.head()


# ## Analisis para base de datos equilibrada de 100 elementos por etiqueta

# In[24]:


Xeq = DFequal2.iloc[:,0:96].values
#X = l.iloc[:,:3].values
yeq = DFequal2.iloc[:,96].values
#y = l.iloc[:,0].values


Xeq_norm=(Xeq-Xeq.min())/(Xeq.max()-Xeq.min())

Xeq_std = StandardScaler().fit_transform(Xeq_norm)
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

plt.figure(figsize=(8, 6))
for lab, col in zip(('RRL','AGN-I','Blazar','CV/Nova','EBC','EBSD/D','LPV','Periodic-Other','Ceph','DSCT','SNII','SNIa'),
                        #('RRL', 'EBC', 'EBSD/D','AGN-I','Blazar','CV/Nova')
                        ('magenta', 'cyan', 'limegreen','blue','yellow','red','brown','orange','gray','black','aqua','pink')):
        plt.scatter(Yeq[yeq==lab, 0],
                    Yeq[yeq==lab, 1],
                    label=lab,
                    c=col)
plt.xlabel('Comp Principal 1 N-Samples')
plt.ylabel('Comp Principal 2 Amplitude-1')
plt.title('Aplicacion PCA')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# In[25]:


DFequal2.iloc[0,0:96]


# In[27]:


matrix_w = np.hstack((eig_pairs[1][1].reshape(96,1),
                      eig_pairs[2][1].reshape(96,1)))

#print('Matriz W:\n', matrix_w)

Yeq = Xeq_std.dot(matrix_w)

plt.figure(figsize=(8, 6))
for lab, col in zip(('RRL','AGN-I','Blazar','CV/Nova','EBC','EBSD/D','LPV','Periodic-Other','Ceph','DSCT','SNII','SNIa'),
                        #('RRL', 'EBC', 'EBSD/D','AGN-I','Blazar','CV/Nova')
                        ('magenta', 'cyan', 'limegreen','blue','yellow','red','brown','orange','gray','black','aqua','pink')):
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


# In[28]:


matrix_w = np.hstack((eig_pairs[0][1].reshape(96,1),
                      eig_pairs[2][1].reshape(96,1)))

#print('Matriz W:\n', matrix_w)

Yeq = Xeq_std.dot(matrix_w)

plt.figure(figsize=(8, 6))
for lab, col in zip(('RRL','AGN-I','Blazar','CV/Nova','EBC','EBSD/D','LPV','Periodic-Other','Ceph','DSCT','SNII','SNIa'),
                        #('RRL', 'EBC', 'EBSD/D','AGN-I','Blazar','CV/Nova')
                        ('magenta', 'cyan', 'limegreen','blue','yellow','red','brown','orange','gray','black','aqua','pink')):
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


# ## Comparacion de caracteristicas

# In[31]:


yeq = DFequal.iloc[:,96].values
means2 = DFequal[['Mean_1', 'Mean_2']].copy()
DFequal.dropna(inplace=True)
plt.figure(figsize=(8, 6))
for lab, col in zip(('RRL', 'EBC', 'EBSD/D','AGN-I','CV/Nova','Blazar','LPV','Periodic-Other'),
                        ('magenta', 'cyan', 'limegreen','blue','r','y','k','g')):
    plt.scatter(
                means2.values[yeq==lab, 0] - means2.values[yeq==lab, 1],
                means2.values[yeq==lab, 0],label=lab,c=col, alpha=0.5)
plt.ylabel('Mean_1')
plt.xlabel('Mean_1 - Mean_2')
plt.legend(loc='lower left')
plt.title('g mean magnitude vs "color" for ALeRCE - ZTF dataset')


# In[132]:


means2 = DFequal[['Mean_1', 'Mean_2']].copy()
means2.dropna(inplace=True)
plt.figure(figsize=(8, 6))
plt.scatter(
    means2.values[:, 0] - means2.values[:, 1],
    means2.values[:, 0],label=lab,c=col)#,alpha=0.005)
plt.ylabel('Mean_1')
plt.xlabel('Mean_1 - Mean_2')
plt.legend()
plt.title('g mean magnitude vs "color" for ALeRCE - ZTF dataset')


# In[157]:


yeq = DFequal.iloc[:,96].values
means2 = DFequal[['Mean_1', 'Mean_2']].copy()
DFequal.dropna(inplace=True)
plt.figure(figsize=(8, 6))
for lab, col in zip(('RRL', 'EBC', 'EBSD/D','AGN-I','CV/Nova','Blazar','LPV','Periodic-Other'),
                        ('magenta', 'cyan', 'limegreen','blue','r','y','k','g')):
    plt.scatter(
                means2.values[yeq==lab, 0] - means2.values[yeq==lab, 1],
                means2.values[yeq==lab, 0],label=lab,c=col, alpha=0.2)
plt.ylabel('Mean_1')
plt.xlabel('Mean_1 - Mean_2')
plt.legend(loc='lower left')
plt.title('g mean magnitude vs "color" for ALeRCE - ZTF dataset')

