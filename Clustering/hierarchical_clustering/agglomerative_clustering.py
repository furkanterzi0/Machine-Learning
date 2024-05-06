# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:04:37 2024

@author: terzi
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('musteriler.csv')

x = data.iloc[:,3:].values

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=4,metric='euclidean',linkage='ward')
'''
metric = Metric used to compute the linkage. Can be “euclidean”, 
“l1”, “l2”, “manhattan”, “cosine”, or “precomputed”. 
!!! If linkage is “ward”, only “euclidean” is accepted. 
If “precomputed”,a distance matrix is needed as input for the fit method.

linkage{‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
‘ward’ minimizes the variance of the clusters being merged.

‘average’ uses the average of the distances of each observation of the two sets.

‘complete’ or ‘maximum’ linkage uses the maximum distances between all observations of the two sets.

‘single’ uses the minimum of the distances between all observations of the two sets.
'''

y_pred = ac.fit_predict(x)

plt.title("HC")
plt.scatter(x[y_pred==0,0], x[y_pred==0,1],s=100,color='red')
# x[y_pred==0,0] = y_predi 0 olan dizinin 0. indexini al
plt.scatter(x[y_pred==1,0], x[y_pred==1,1],s=100,color='blue')
# x[y_pred==1,0]= y_predi 1 olan dizinin 0. indexini al
plt.scatter(x[y_pred==2,0], x[y_pred==2,1],s=100,color='green')
plt.scatter(x[y_pred==3,0], x[y_pred==3,1],s=100,color='yellow')
plt.show()

#dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))

