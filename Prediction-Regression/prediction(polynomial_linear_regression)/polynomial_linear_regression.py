# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:01:18 2024

@author: terzi
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
veriler = pd.read_csv('maaslar.csv')

x = veriler[['Egitim Seviyesi']]
y = veriler[['maas']]
X = x.values
Y = y.values

# linear regression
lin_reg = LinearRegression().fit(X,Y)

plt.scatter(X,Y, color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.show()

# polynomial regression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2) #2. dereceden polynomial obje olustur
x_poly = poly_reg.fit_transform(X) # X değerini polinomal dünyaya çevirdi
print(x_poly)

lin_reg2 = lin_reg = LinearRegression().fit(x_poly,Y) # 3 tane kolonun carpanlarını öğreniyo
                                            #x^0 x^1 x^2

plt.scatter(X,Y, color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')



