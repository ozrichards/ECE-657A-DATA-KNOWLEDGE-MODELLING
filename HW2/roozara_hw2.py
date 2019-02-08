#!/usr/bin/env python
# coding: utf-8

# -  __ECE 657A: Data and Knowledge Modelling and Analysis__
# - __Winter 2019__
# - __WATIAM:roozara ID: 20801583__
# - __Homework 2:Data Normalization and .....__
# 
# Reference used : [About Feature Scaling and Normalization , Sebastian Raschka](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html#z-score-standardization-or-min-max-scaling) 

# In[24]:


import pandas as pd
import numpy as np
#calculate zsore normalization and min-max normalizatio
from sklearn import preprocessing
#calculate the distance between datapoints
from scipy.spatial import distance


#importing the red-wine dataset into variable wd
wd = pd.read_csv('data/winequality-red.csv',sep= ';') 
wd = wd.iloc[:10,:]
print(wd)



# # Calculate the min-max normalized values

# In[25]:



#min-max normalized values calculating manually

#wd_norm= (wd-wd.min())/(wd.max() - wd.min())
#print(wd_norm)

#min-max normalized values calculating manually
minmax_scale = preprocessing.MinMaxScaler().fit(wd.iloc[:,:])
wd_minmax = minmax_scale.transform(wd.iloc[:,:])
wd_minmax = pd.DataFrame(wd_minmax)
wd_minmax.columns = wd.columns
print(wd_minmax)




# # Calculate the Z-score normalized values

# In[26]:


#z-score normalized values
std_scale = preprocessing.StandardScaler().fit(wd.iloc[:,:])
wd_z = std_scale.transform(wd.iloc[:,:])
wd_z=pd.DataFrame(wd_z)
wd_z.columns =wd.columns
print(wd_z)



# # Calculate the mean subtracted normalized values

# In[27]:


# mean subtracted normalized values
wd_mean= wd.mean()
wd_meansub_norm = wd - wd.mean() 
print (wd_meansub_norm )
 


# # Calculate Manhatten distance for each of the first 10 points

# In[28]:




d_matrix= distance.squareform(distance.pdist(wd,'cityblock'))
d_matrix= pd.DataFrame(d_matrix)
np.fill_diagonal(d_matrix.values, 'nan')
print(d_matrix)


# In[29]:


man_dist=pd.DataFrame([d_matrix.idxmin(axis=0),d_matrix.min(axis=0),d_matrix.idxmax(axis=0),d_matrix.max(axis=0)]).T
man_dist.columns=["index dest","nearest(min)","index dest","farthest(max)"]
man_dist.index.name = ['index source']
print(man_dist)


# # Calculate Euclidean distance for each of the first 10 points

# In[30]:



d_eucl= distance.squareform(distance.pdist(wd,'euclidean'))
d_eucl= pd.DataFrame(d_eucl)
np.fill_diagonal(d_eucl.values, 'nan')
print(d_eucl)


# In[31]:


eucl_dist=pd.DataFrame([d_eucl.idxmin(axis=0),d_eucl.min(axis=0),d_eucl.idxmax(axis=0),d_eucl.max(axis=0)]).T
eucl_dist.columns=["index dest","nearest(min)","index dest","farthest(max)"]
eucl_dist.index.name = ['index source']
print(eucl_dist)


# # Calculate cosine distance for each of the first 10 points

# In[32]:


d_cos= distance.squareform(distance.pdist(wd,'cosine'))
d_cos= pd.DataFrame(d_cos)
np.fill_diagonal(d_cos.values, 'nan')
print(d_cos)


# In[33]:


cos_dist=pd.DataFrame([d_cos.idxmin(axis=0),d_cos.min(axis=0),d_cos.idxmax(axis=0),d_cos.max(axis=0)]).T
cos_dist.columns=["index dest","nearest(min)","index dest","farthest(max)"]
cos_dist.index.name = ['index source']
print(cos_dist)

