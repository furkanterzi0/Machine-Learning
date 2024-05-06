# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:52:25 2024

@author: terzi
"""
# map(function, iterable)

import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score

data = pd.read_csv('data.csv')

x = data.iloc[:,:3].values
y = data.iloc[:,-1:].values
y = y.ravel() 


x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.30,random_state=2)

sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test) 

svc = SVC(kernel='poly')
svc.fit(x_train_sc, y_train)

y_pred = svc.predict(x_test_sc)


cm = confusion_matrix(y_test, y_pred)
print(cm)

print("accuracy score : ", accuracy_score(y_test, y_pred))

