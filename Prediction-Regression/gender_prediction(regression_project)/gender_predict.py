# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:07:49 2024

@author: terzi
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm

original_data = pd.read_csv('data.csv') 

gender = original_data[['Gender']]

gender_le = LabelEncoder().fit_transform(gender)

x = original_data.iloc[:,:3].values
y = pd.DataFrame(data=gender_le,index=range(40),columns=['gender']).values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30,random_state=0)

y_pred = LinearRegression().fit(x_train, y_train).predict(x_test)


# SVR 

svr_reg = SVR(kernel='rbf',degree=3)

scaler_x = StandardScaler() # !!!!

x_train_sc = scaler_x.fit_transform(x_train)
x_test_sc = scaler_x.transform(x_test)


svr_reg.fit(x_train_sc, y_train)
y_pred_svr = svr_reg.predict(x_test_sc)


'''
print("PREDİCT: \n SVR -rbf regression predict")

count=0
for item in y_pred_svr:
    
    if(item <= 0):
        print("-1\t\t\t",end="")
    else:
        print("1\t\t\t",end="")
    
    print(y_test_sc[count])
    
    count+=1
''' 

plt.plot(y_test)
plt.plot(y_pred_svr)
plt.show()


Height = float(input("Height(cm): "))
Weight = float(input("Weight(kg): "))
Age = int(input("Age: "))

user_data = pd.DataFrame({'Height': [Height], 'Weight': [Weight], 'Age': [Age]})

user_data_sc = scaler_x.transform(user_data) # !


user_pred = svr_reg.predict(user_data_sc)

print("Predicted gender: ", "Female" if user_pred <= 0.5 else "Male")

print(f'R2 SCORE DEGERİ : {r2_score(y_test, y_pred)}')



model = sm.OLS(y_pred_svr, y_test)
print(model.fit().summary()) # R-squared (uncentered): 0.861


        
