#!/usr/bin/env python
# coding: utf-8

# -  __ECE 657A: Data and Knowledge Modelling and Analysis__
# - __Winter 2019__
# - __WATIAM:roozara ID: 20801583__
# - __Homework 3:Eigenvector Decomposition__
# 
# Reference used :https://hadrienj.github.io/posts/Preprocessing-for-deep-learning/
# https://archive.ics.uci.edu/ml/machine-learning-databases/communities/

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#for eigen value
from numpy import cov
from numpy import linalg as LA
#pd.options.display.max_columns = None
pd.options.display.max_rows = None

def readFromFile(name):
    with open(name) as f: 
         features= [line.split(' ')[1] for line in f.readlines()
                    if line.startswith('@attr')]
    return features  
                


 
   


# # Importing the crime dataset and storing in a matrix
# The crime dataset was loaded into a matrix ,its observed that the data is already normalized but so many missing values.
# About 1675 out of 1993 missing values each in columns 101 to 126 were replaced by the mean of those features. the first
# 5 columns including two (county & community of which contained about 1174 missing values each were not included in 
# creating the matrix.These first five non predictive attributes were left out of the analysis. 128 scaled to 123 features
# being analysed.      

# In[67]:


#importing the communities dataset into variable cdata
cdata = pd.read_csv('data/communities.data',sep= ',', header = None, na_values=["?"]) 



#print(Acdata)
#examining the data to correct for missing valiue

#IDENTIFYING COLUMNS WITH null value and filling with the mean
#nan_col=pd.DataFrame(cdata.isnull().sum(axis=0))
#nan_col.index =  list(range(128))

cdata.iloc[:,4:] = cdata.iloc[:,4:].apply(lambda x: x.fillna(x.mean()),axis=0)
cdata.columns = readFromFile('data/communities.names')
print('Crime dataset dimension', cdata.shape)
cdata.head(5)


# In[66]:


#matrix creating with numpy
cdata_matrix = np.matrix(cdata.iloc[:,5:])
print('Matrix created' ,'(dimension' ,cdata_matrix.shape,')','\n',cdata_matrix)


# # Compute the eigenvectors and eigenvalue and Reporting the top 20 eigenvalues
# We compute the eigenvectors and thus eigen value by first calculating the covaraince matrix. We project
# any data onto the principal subspace that is spanned by the eigenvectors that belong to the largest eigenvalues.

# In[46]:



cov_matrix = np.cov(cdata_matrix, rowvar=False, bias=True)
eigenvalues, eigenVector = LA.eig(cov_matrix)
x = np.arange(1, 124,1)
eig_valTable = pd.DataFrame(eigenvalues, index = x, columns = ['Eigenvalues'])
eig_valTable.sort_values(by='Eigenvalues', ascending=False, inplace=True)
print(eig_valTable.head(20))


# As it can be seen from the first plot (left plot) below, it is hard to have a clear cut off since the
# curve is more shallow. The first 20 eigenvalues count for ~85% of the variance. The 95% were
# calculated below and it turned out that we need approximately 39 eigenvalues to process 95% of
# the data.

# In[61]:


i = 1
j = 0  
y = np.zeros(shape=(123,1))

for index, row in eig_valTable.iterrows():
    y[j] = eig_valTable['Eigenvalues'][i] + y[j-1]
    i += 1
    j += 1 
sum_eigen = pd.DataFrame(y, index = x)
sum_eigen['Eigenvalue No'] = x   
    
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x[:21], y[:21])
ax1.set_title('plot showing\n20 eigenvalues')
ax1.set_xlabel('Eigenvalues')
ax1.set_ylabel('sum(Eigenvalues)')

ax2.plot(x, y)
ax2.set_title(' plot showing \nall eigenvalues')
ax2.set_xlabel('Eigenvalues')
ax2.set_ylabel('sum(Eigenvalues)')


# In[59]:


print(sum_eigen[(sum_eigen[0] > 3.98) & (sum_eigen[0] < 3.992268253580597)])

