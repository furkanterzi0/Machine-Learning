# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:47:50 2024

@author: terzi
"""

import pandas as pd
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('satislar.csv')


aylar = data[['Aylar']]
satislar = data[['Satislar']]

x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

'''
X_train = StandardScaler().fit_transform(x_train)
X_test = StandardScaler().fit_transform(x_test)

Y_train = StandardScaler().fit_transform(y_train)
Y_test = StandardScaler().fit_transform(y_test)
''' # suan kullandıgımız modelde ölcekleme yapmaya ihtiyac yok bazi algoritmalarda ihtiyac var

#model insasi (linear regression)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train) # x_train ile y_train degerlerini öğreniyo x->y

#tahmin = LinearRegression().fit(X_train, Y_train).predict(X_test)
tahmin = lr.predict(x_test)         #predict -> tahmin  
                                    # X_test amacı x i kullanarak Y_testi bulabilcek mi kontrol için
# x train ve y train ile eğittikten sonra test icin ayrılan veriyi predict fonksiyonuna gönderip
# test verisi üzerinden gercek veriyi[Y_test] tahmin ettiriyoruz
                                   
# veri gorsellestirme
import matplotlib.pyplot as plt

x_train = x_train.sort_index() # indexe göre siraliyo
y_train = y_train.sort_index() # indexe göre siraliyo

plt.plot(x_train, y_train) # plot-> grafigini cizmek
plt.plot(x_test, lr.predict(x_test)) # x_test'i kullanarak tahmin et

plt.title('aylara göre satış')
plt.xlabel('aylar')
plt.ylabel('satışlar')




