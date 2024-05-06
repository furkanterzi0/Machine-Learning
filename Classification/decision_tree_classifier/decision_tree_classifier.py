# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:51:21 2024

@author: terzi
"""

import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

data = pd.read_csv('data.csv')

x = data.iloc[:,:3].values
y = data.iloc[:,-1:].values

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.30,random_state=2)

sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test) 

dtc = DecisionTreeClassifier(criterion='entropy') # or gini = formulun log tabanı farklı
dtc.fit(x_train_sc, y_train)

y_pred = dtc.predict(x_test_sc)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n",cm)

print("accuracy score : ", accuracy_score(y_test, y_pred))


