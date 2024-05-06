# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 00:52:33 2024

@author: terzi
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

data = pd.read_csv('HousingData.csv')
impute_data = KNNImputer(missing_values=np.nan,n_neighbors=5).fit_transform(data)

x = impute_data[:,:-1]
y = impute_data[:,-1:]
y=np.ravel(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=3)

sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

svr_reg = SVR(kernel='rbf',degree=13)
svr_reg.fit(x_train_sc, y_train)

y_pred = svr_reg.predict(x_test_sc)

print(r2_score(y_test, y_pred))

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

