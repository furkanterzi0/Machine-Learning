# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 18:13:35 2024

@author: terzi
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

data = pd.read_excel('Iris.xlsx')

x = data.iloc[:,:-1].values
y = data.iloc[:,-1:].values
y = np.ravel(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=3)

sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)


# logistic function
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=2) 
log_reg.fit(x_train_sc, y_train)

y_pred_logistic = log_reg.predict(x_test_sc)

print("Logistic\n",confusion_matrix(y_test, y_pred_logistic))
print("Logistic Function accuracy score : ", accuracy_score(y_test, y_pred_logistic),"\n")


# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(x_train_sc, y_train)

y_pred_knn = knn.predict(x_test_sc)

print("KNN\n",confusion_matrix(y_test, y_pred_knn))
print("KNN accuracy score : ", accuracy_score(y_test, y_pred_knn),"\n")


# SVC
from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(x_train_sc, y_train)

y_pred_svc = svc.predict(x_test_sc)

print("SVC\n",confusion_matrix(y_test, y_pred_svc))
print("SVC accuracy score : ", accuracy_score(y_test, y_pred_svc),"\n")


# Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train_sc, y_train)

y_pred_nb = gnb.predict(x_test_sc)

print("Naive Bayes\n",confusion_matrix(y_test, y_pred_nb))
print("Naive Bayes accuracy score : ", accuracy_score(y_test, y_pred_nb),"\n")


# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(x_train_sc, y_train)

y_pred_dt = dtc.predict(x_test_sc)

print("Decision Tree\n",confusion_matrix(y_test, y_pred_dt))
print("Decision Tree accuracy score : ", accuracy_score(y_test, y_pred_dt),"\n")


# Random Forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50,criterion='entropy')
rfc.fit(x_train_sc,y_train)

y_pred_rf = rfc.predict(x_test_sc)

print("Random Forest\n",confusion_matrix(y_test, y_pred_rf))
print("Random Forest accuracy score : ", accuracy_score(y_test, y_pred_rf),"\n")

