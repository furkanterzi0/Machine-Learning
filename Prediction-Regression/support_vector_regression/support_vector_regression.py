# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 23:08:27 2024

@author: terzi
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
import matplotlib.pyplot as plt

original_data = pd.read_csv('maaslar.csv')

x = original_data.iloc[:,1:2].values
y = original_data.iloc[:,-1:].values


poly_reg = PolynomialFeatures(degree=4)
lin_reg = LinearRegression().fit(poly_reg.fit_transform(x), y)


predict = lin_reg.predict(poly_reg.fit_transform(x))

plt.scatter(x,y, color='red')
plt.plot(x,predict,color='green')
plt.show()

# support vector regression scaler ile kullanmamız gerekli veriler uzerindeki marjinal verilere duyarlı

x_sc = StandardScaler().fit_transform(x)
y_sc = StandardScaler().fit_transform(y)


# SVR        svm -> support vector machine
from sklearn.svm import SVR

svr_reg = SVR(kernel="rbf") # kernel = 'poly'  || kernel = 'linear'
svr_reg.fit(x_sc , y_sc)

plt.scatter(x_sc , y_sc)
plt.plot(x_sc, svr_reg.predict(x_sc))
plt.show()



