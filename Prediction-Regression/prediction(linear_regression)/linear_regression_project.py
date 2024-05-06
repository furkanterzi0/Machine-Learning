# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:00:35 2024

@author: terzi
"""
#1 Veri setini kullanarak bir doğrusal regresyon modeli oluşturun.
#2 Modelinizi eğitmek için veri setinin bir kısmını (örneğin, %80'i) kullanın ve geri kalanını test etmek için kullanın.
#3 Modelinizi test edin ve tahminlerin gerçek değerlerle ne kadar uyumlu olduğunu değerlendirin.
#4 Oluşturduğunuz modeli kullanarak bir grafik çizin. Grafikte, orijinal veri noktalarını ve doğrusal regresyon çizgisini gösterin.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('linear_regression_data.csv')

x = data[['X']]
y = data[['Y']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

lr = LinearRegression().fit(x_train, y_train)

predict = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.scatter(x_train, y_train, color='blue', label='Eğitim Verisi') # scatter -> dagılım
plt.scatter(x_test, y_test, color='red', label='Test Verisi')
plt.plot(x_test, predict, color='green', label='Doğrusal Regresyon Çizgisi')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Doğrusal Regresyon')
plt.legend()
plt.show()