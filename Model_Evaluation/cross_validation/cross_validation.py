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

X = data.iloc[:,:-1].values
y = data.iloc[:,-1:].values
y = np.ravel(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=1)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

bnb = BernoulliNB()
bnb.fit(X_train_sc,y_train)

y_pred = bnb.predict(X_test_sc)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print("accuracy score : ",accuracy_score(y_test, y_pred))

#cross validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True) # shuffle -> True veriler ayırılırken karıştırarak ayırma

from sklearn.model_selection import cross_val_score
#estimator = kullanılan model neyse logistic regression,naive bayes,random forest [degisken ismi]
# su anlik bernoulli nb -> bnb          estimator -> tahminci
scores = cross_val_score(estimator = bnb, X=X_train, y=y_train, cv=kf)
'''
 cv: cv parametresi, bu katlama sayısını belirtir. Örneğin, cv=5 ise,
 veri seti beş alt kümeye bölünecek ve çapraz doğrulama beş katlamada gerçekleşecektir.
 Bu, modelin beş farklı eğitim ve test seti kombinasyonunda değerlendirileceği anlamına gelir.
 
'''

print("Her bir katmanın performansı:", scores)
print("Ortalama performans:", scores.mean())
print("Max performans:", scores.max())


