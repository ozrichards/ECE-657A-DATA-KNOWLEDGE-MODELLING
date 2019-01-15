#Import pandas package as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
data = pd.read_csv(file_url,sep= ',',header = None)
#data = data.iloc[:,:32]
#create the features columns
features = ['ID number', 'Diagnosis', 'mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean compactness',
            'mean concavity','mean concave points','mean symmetry','mean fractal dimension', 'SE radius','SE texture','SE perimeter',
            'SE area','SE smoothness','SE compactness','SE concavity', 'SE concave points','SE symmetry', 'SE fractal dimension',
            'Worst radius', 'Worst texture', 'Worst perimeter', 'Worst area','Worst smoothness', 'Worst compactness','Worst concavity',
            'Worst concave points','Worst symmetry','Worst fractal dimension']

data.columns = features
data = data.set_index('ID number')
print ("****************This is  the cancer dataset showing first five rows only**************************")  
print(data.head())

#calculate mean
print ("*****************This is Mean of the cancer dataset*************************")      
mean_results = data.loc[:,features[2:32]].mean()
print (mean_results)

#calc mode
print ("*****************This is Mode of the cancer dataset*************************")  
mode_results=data.loc[:,features[2:32]].mode().iloc[0,:]
print(mode_results)

#calc skew
print ("*****************This is Skew of the cancer dataset*************************")  
skew_results = data.loc[:,features[2:32]].skew()
print(skew_results)

#calc standard deviation
print ("*****************This is Standard Deviation of the cancer dataset*************************")  
SD_results= data.loc[:,features[2:32]].std()
print(SD_results)

#calc variance
print ("*****************This is Variance of the cancer dataset*************************")  
var_results=data.loc[:,features[2:32]].var()
print(var_results)

#calc correlations using PCC
print ("*****************Correlations*************************")  
correlations = data.drop('Diagnosis', 1).corr(method='pearson')
print(correlations)

correlations_ = correlations.where(np.triu(np.ones(correlations.shape)).astype(np.bool))
correlations_ = correlations_.stack().reset_index()
correlations_.columns = ['By ROW', 'By COLUMN', 'VALUE']
correlations_.loc[correlations_['By ROW'] == correlations_['By COLUMN'], 'VALUE'] = np.nan
correlations_ = correlations_.sort_values(by=['VALUE'], ascending=False).dropna()
print(correlations_)

#pploting histogram 
print ("*****************HISTOGRAM*************************")  
plt.hist= data.hist(column = 'mean perimeter',by = 'Diagnosis')

plt.xlabel('perimeter')
plt.ylabel('count')
plt.show()
