# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:30:59 2024

@author: terzi
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

original_data = pd.read_csv('maaslar.csv')

x = original_data.iloc[:,1:2].values
y = original_data[['maas']].values

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0) # n_estimators -> decision tree sayisi - > tahminciler
rf_reg.fit(x,y) # ravel() -> iki boyutlu diziyi tek boyuta çevirme 

plt.scatter(x,y,color='red')
plt.plot(x,rf_reg.predict(x),color='blue')

z = x + 0.5

plt.plot(x,rf_reg.predict(z),color='green')
plt.show()