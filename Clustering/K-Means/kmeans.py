# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:55:13 2024

@author: terzi
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('musteriler.csv')

x = data.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4,init='k-means++')
kmeans.fit(x)

print(kmeans.cluster_centers_)

sonuclar = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i , init='k-means++', random_state=40)
    kmeans.fit(x)
    sonuclar.append(kmeans.inertia_) # kmeans.inertia_  = wcss

plt.plot(range(1,11),sonuclar)
plt.show()

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=40)
y_pred=kmeans.fit_predict(x)

plt.title('KMeans')
plt.scatter(x[y_pred==0,0], x[y_pred==0,1],s=100,color='red')
# x[y_pred==0,0] = y_predi 0 olan dizinin 0. indexini al
plt.scatter(x[y_pred==1,0], x[y_pred==1,1],s=100,color='blue')
# x[y_pred==1,0]= y_predi 1 olan dizinin 0. indexini al
plt.scatter(x[y_pred==2,0], x[y_pred==2,1],s=100,color='green')
plt.scatter(x[y_pred==3,0], x[y_pred==3,1],s=100,color='yellow')

plt.show()