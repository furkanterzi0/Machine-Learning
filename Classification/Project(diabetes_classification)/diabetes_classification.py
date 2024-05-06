# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 00:08:45 2024

@author: terzi
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix

data = pd.read_csv('diabetes.csv')

x = data.iloc[:,:-1].values
y = data.iloc[:,-1:].values
y = np.ravel(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=1)

sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

bnb = BernoulliNB()
bnb.fit(x_train_sc,y_train)

y_pred = bnb.predict(x_test_sc)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print("accuracy score : ",accuracy_score(y_test, y_pred))