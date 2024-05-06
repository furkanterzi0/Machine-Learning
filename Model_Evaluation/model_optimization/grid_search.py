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

X = data.iloc[:,:3].values
y = data.iloc[:,-1:].values
y = y.ravel() 


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.30,random_state=2)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test) 

svc = SVC(kernel='poly',random_state=0)
svc.fit(X_train_sc, y_train)

y_pred = svc.predict(X_test_sc)


cm = confusion_matrix(y_test, y_pred)
print(cm)

print("accuracy score : ", accuracy_score(y_test, y_pred))


# parametre optimizasyonu
from sklearn.model_selection import GridSearchCV
p= [
    {'C':[1,2,3,4,5], 'kernel':['linear']},
    {'C':[1,10,100,100], 'kernel':['rbf'], 'gamma':[1,0.5,0.1,0.01,0.001]}
   ]
'''
 estimator ->#estimator = kullanılan model neyse logistic regression,naive bayes,random forest [degisken ismi]
 param_grid -> parametreler/denenecekler
 scoring: neye göre skorlanacak : örn:accuracy

 cv: cv parametresi, bu katlama sayısını belirtir. Örneğin, cv=5 ise,
 veri seti beş alt kümeye bölünecek ve çapraz doğrulama beş katlamada gerçekleşecektir.
 Bu, modelin beş farklı eğitim ve test seti kombinasyonunda değerlendirileceği anlamına gelir.

 n_jobs = aynı anda çalışacak iş
 
'''
gs = GridSearchCV(estimator=svc, param_grid=p,scoring='accuracy', cv=5, n_jobs=-1)
'''
n_jobs=-1 parametresi, scikit-learn kütüphanesindeki GridSearchCV 
fonksiyonunda, hesaplamanın mevcut makinedeki tüm işlemciler üzerinde 
dağıtılacağını belirtir. Yani, grid aramasının işlemi, birden fazla işlemciyi 
kullanarak daha hızlı hale getirmek için tüm CPU çekirdeklerini kullanır.

Eğer n_jobs=1 olarak ayarlarsanız, hesaplama tek bir işlemci çekirdeğinde
gerçekleştirilir. Bu, paralel hesaplama yapılmayacağı anlamına gelir ve işlemin tek bir
CPU çekirdeğinde sıralı olarak çalışacağı anlamına gelir. Bu, bazı durumlarda performans 
açısından dezavantajlı olabilir, ancak bazı sistemlerde uyumluluk sorunlarını çözebilir.
'''

grid_search = gs.fit(X_train,y_train)

print('en iyi sonuc: ', grid_search.best_score_)
print('en iyi parametreler: ', grid_search.best_params_)

'''
accuracy score :  0.8571428571428571[before optimization]
en iyi sonuc:  0.9333333333333332 [after]
en iyi parametreler:  {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
'''

