# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:16:19 2024

@author: terzi
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('data.csv')

x = data.iloc[:,:3].values
y = data.iloc[:,-1:].values
y = y.ravel() #Bu, y değişkenini beklenen 1D formata getirecek ve uyarıyı ortadan kaldıracaktır.

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.30,random_state=2)


knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
        
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)


import pickle
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn, file)
