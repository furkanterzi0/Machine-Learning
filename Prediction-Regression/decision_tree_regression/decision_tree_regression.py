# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 00:48:33 2024

@author: terzi
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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

# decision tree
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(random_state=0)

dt_reg.fit(x, y)

plt.scatter(x , y , color = 'red')
plt.plot(x , dt_reg.predict(x) , color='blue')
plt.show()

print(dt_reg.predict([[5.6]]))
print(dt_reg.predict([[6]])) 
print(dt_reg.predict([[6.5]]))
