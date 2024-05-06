# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:14:59 2024

@author: terzi
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

original_data = pd.read_csv('tenis.csv')

outlook = original_data[['outlook']]
windy = original_data[['windy']]

play = original_data[['play']]


outlook_encoded = OneHotEncoder().fit_transform(outlook).toarray()
windy_encoded = LabelEncoder().fit_transform(windy)
play_encoded = LabelEncoder().fit_transform(play)


outlook_df = pd.DataFrame(data=outlook_encoded, index=range(14),columns=['overcast','rainy','sunny'])
windy_df = pd.DataFrame(data=windy_encoded, index=range(14),columns=['windy'])
play_df = pd.DataFrame(data=play_encoded, index=range(14),columns=['play'])


data = original_data.iloc[:,1:3]

data = pd.concat([outlook_df,data],axis=1)
data = pd.concat([data,windy_df],axis=1)

x_train, x_test, y_train, y_test = train_test_split(data,play_df,test_size=0.22, random_state=2)

regressor = LinearRegression().fit(x_train,y_train)

y_pred = regressor.predict(x_test)  
print(y_pred)
print(y_test)
#stat

X = np.append(arr=np.ones((14,1)).astype(int),  values=data, axis=1)


X_list = data.iloc[:,[0,1,2,5]].values 
#[0,1,2,3,4,5] yapıldıgında 3 ve 4 indisli stuunlar[teker teker kontrol edildi] 
#p degeri cok yuksek çıktıgı için sistemden çıkartıldı

model = sm.OLS(play_df, X_list).fit()
print(model.summary())


x_train = x_train.iloc[:,[0,1,2,5]]
x_test = x_test.iloc[:,[0,1,2,5]]


regressor = LinearRegression().fit(x_train,y_train)
y_pred = regressor.predict(x_test)  

print(y_pred)
print(y_test)
