# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:55:07 2024

@author: terzi
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

data = pd.read_csv('data1.csv') # x^2 + 1 fonksiyonu    - > 1-> 2 , 2->5 , 3->10

x = data[['x']]
y = data[['y']]
X = x.values
Y = y.values


poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)

lin_reg = LinearRegression().fit(x_poly, y)

plt.scatter(X, Y, color='red', label='Eğitim Verisi')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color='blue',label='Tahmin Polinomu')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()

