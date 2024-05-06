# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 01:58:05 2024

@author: terzi
"""

# naive bayes
# logistic
# rassal orman
# genderi labelencode yap
# age ve embarekd impute

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score

data = pd.read_csv('train.csv')

sex_encode = LabelEncoder().fit_transform(data['Sex'])

age_impute = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data[['Age']])

embarked_impute = SimpleImputer(strategy='most_frequent').fit_transform(data[['Embarked']])
embarked_ohe = OneHotEncoder().fit_transform(embarked_impute)

x = data.drop(columns=["PassengerId","Survived","Name","Cabin","Ticket","Sex","Embarked","Age"])
x['Age'] = age_impute
x['Sex'] = sex_encode
x = pd.concat([x, pd.DataFrame(embarked_ohe.toarray(), columns=["Embarked_" + str(int(i)) for i in range(embarked_ohe.shape[1])])], axis=1)

y = data[["Survived"]]
y = np.ravel(y)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=6)

sc = StandardScaler()
sc.fit(x_train)

x_train_sc = sc.transform(x_train)
x_test_sc = sc.transform(x_test)

from sklearn.naive_bayes import BernoulliNB 
bnb = BernoulliNB().fit(x_train_sc, y_train)
y_pred_nb = bnb.predict(x_test_sc)

print("naive bayes \n",confusion_matrix(y_test, y_pred_nb))
print("accuracy score : ", accuracy_score(y_test, y_pred_nb))

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42) 
log_reg.fit(x_train_sc, y_train)

y_pred_lg = log_reg.predict(x_test_sc)

print("logistic regression \n",confusion_matrix(y_test, y_pred_lg))
print("accuracy score : ", accuracy_score(y_test, y_pred_lg))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50,criterion='entropy')
rfc.fit(x_train_sc,y_train)

y_pred_rfc = rfc.predict(x_test_sc)

print("Random Forest \n",confusion_matrix(y_test, y_pred_rfc))
print("accuracy score : ", accuracy_score(y_test, y_pred_rfc))

pclass = int(input("pclass: "))
sibsp = int(input("sibsp: "))
parch = int(input("parch: "))
fare = float(input("fare: "))
age = int(input("age: "))
sex = int(input("sex (0 for female, 1 for male): "))
embarked = input("embarked (s for Southampton, q for Queenstown, c for Cherbourg): ")

if embarked == 's':
    embarked = [0,0,1]
elif embarked == 'q':
    embarked = [0,1,0]
else:
    embarked = [1,0,0]

person = [pclass, sibsp, parch, fare, age, sex] + embarked

person = np.array(person).reshape(1, -1)

person_sc = sc.transform(person)

y_pred_survive = log_reg.predict(person_sc)
print("Predicted Survived:", y_pred_survive)



import warnings
warnings.filterwarnings("ignore", category=UserWarning)



