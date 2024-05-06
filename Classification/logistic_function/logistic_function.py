# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 03:34:32 2024

@author: terzi
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score

data = pd.read_csv('data.csv')

x = data.iloc[:,:3].values
y = data.iloc[:,-1:].values

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.30,random_state=2)

x_train = pd.DataFrame(x_train, columns=data.columns[:-1])
x_test = pd.DataFrame(x_test, columns=data.columns[:-1])

sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test) 

log_reg = LogisticRegression(random_state=0) 
log_reg.fit(x_train_sc, y_train)

y_pred = log_reg.predict(x_test_sc)

cm = confusion_matrix(y_test, y_pred)
print(cm)

Height = float(input("Height(cm): "))
Weight = float(input("Weight(kg): "))
Age = int(input("Age: "))

user_data = pd.DataFrame({'Height': [Height], 'Weight': [Weight], 'Age': [Age]})

user_data_sc = sc.transform(user_data) # !


user_pred = log_reg.predict(user_data_sc)

print("Predicted gender: ", "Female" if user_pred =="Female" else "Male")

print("accuracy score : ", accuracy_score(y_test, y_pred))

