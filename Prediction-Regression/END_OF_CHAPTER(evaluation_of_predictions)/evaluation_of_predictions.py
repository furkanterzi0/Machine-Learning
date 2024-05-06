# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:46:25 2024

@author: terzi
"""

# ÖDEV 
# Gerekli / gereksiz bağımsız değişkenleri bulunuz
# 5 Farklı Yönteme göre regresyon modellerini çıkarınız
#    - MLR, PR, SVR, DT, RF
# Yöntemlerin başarılarını karşılaştırınız
# 10 yıl tecrubeli ve 100 puan almış bir ceo ve aynı özelliklere sahip bir müdürün maaşlarını 
# 5 yöntemle de tahmin edip sonuçları yorumlayın

import pandas as pd
import numpy as np
from sklearn.svm import SVR
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv('maaslar_yeni.csv')

x = data.iloc[:,2:5].values
y = data.iloc[:,-1:].values

test_person_ceo = pd.DataFrame({"Calisan ID": [30], "unvan":['CEO'], "UnvanSeviyesi":[10], "Kidem":[10],"Puan":[100]})
test_person_mudur = pd.DataFrame({"Calisan ID": [31], "unvan":['Mudur'], "UnvanSeviyesi":[7], "Kidem":[10],"Puan":[100]})

test_person_ceo = test_person_ceo.iloc[:,2:5].values
test_person_mudur = test_person_mudur.iloc[:,2:5].values

# ------------------------------------- MLR ----------------------------------

lin_reg = LinearRegression()
lin_reg.fit(x, y)

y_pred_mlr = lin_reg.predict(x)

model = sm.OLS(y_pred_mlr, x)
print(model.fit().summary()) # R-squared (uncentered):  0.903

print("\n\n\n")

print(f'MLR R2 SCORE : {r2_score(y,y_pred_mlr )}') # 0.5857207050854021

print(f'MLR - TEST PERSON CEO MAAS : { lin_reg.predict(test_person_ceo) }') # 32861.59416921
print(f'MLR - TEST PERSON MUDUR MAAS : { lin_reg.predict(test_person_mudur) }') # 22565.31899734

print("\n\n\n\n\n\n")
# -------------------------------------- PR -----------------------------------

poly_reg = PolynomialFeatures(degree=2)

x_poly = poly_reg.fit_transform(x)
ceo_poly = poly_reg.fit_transform(test_person_ceo)
mudur_poly = poly_reg.fit_transform(test_person_mudur)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

y_pred_pr = lin_reg2.predict(x_poly)

model = sm.OLS(y_pred_pr, x_poly)
print(model.fit().summary()) #  R-squared:  1.000

print("\n\n\n")

print(f'PR R2 SCORE : {r2_score(y, y_pred_pr )}') # 0.8871429857650684

print(f'PR - TEST PERSON CEO MAAS : { lin_reg2.predict(ceo_poly) }') # 61096.27264515
print(f'PR - TEST PERSON MUDUR MAAS : { lin_reg2.predict(mudur_poly) }') # 24491.01611224

print("\n\n\n\n\n\n")

# ------------------------------------- SVR ----------------------------------

scaler = StandardScaler()
scaler.fit(x)

x_sc = scaler.transform(x)

y_scaler = StandardScaler()
y_sc = y_scaler.fit_transform(y)

ceo_sc = scaler.transform(test_person_ceo)
mudur_sc = scaler.transform(test_person_mudur)

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_sc,y_sc)

y_pred_svr = svr_reg.predict(x_sc)

model = sm.OLS(y_pred_svr, x_sc)
print(model.fit().summary()) # R-squared (uncentered):  0.782

print("\n\n\n")

print(f'SVR R2 SCORE : { r2_score(y_sc, y_pred_svr ) }') # 0.6287203839391851

# CEO maas tahminini geri dönüştürme
ceo_maas_pred = svr_reg.predict(ceo_sc.reshape(1, -1))
ceo_maas_pred_2d = np.array([ceo_maas_pred])  # 1D diziyi 2D diziye dönüştürme
ceo_maas_pred_inverse = y_scaler.inverse_transform(ceo_maas_pred_2d)

# Mudur maas tahminini geri dönüştürme
mudur_maas_pred = svr_reg.predict(mudur_sc.reshape(1, -1))
mudur_maas_pred_2d = np.array([mudur_maas_pred])  # 1D diziyi 2D diziye dönüştürme
mudur_maas_pred_inverse = y_scaler.inverse_transform(mudur_maas_pred_2d)

print(f'SVR - TEST PERSON CEO MAAS : {ceo_maas_pred_inverse}') # 19877.10775064
print(f'SVR - TEST PERSON MUDUR MAAS : {mudur_maas_pred_inverse}') # 12779.99798041

print("\n\n\n\n\n\n")

# -------------------------------------- DT -----------------------------------

dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x, y)

y_pred_dt = dt_reg.predict(x)

model = sm.OLS(y_pred_dt, x)
print(model.fit().summary()) # R-squared (uncentered):  0.679

#print(f'DT R2 SCORE : {r2_score(y, y_pred_dt )}') # DT'DE R2 SCORE YANILTICI  [1.0]
# Adj. R-squared (uncentered):              0.644

print("\n\n\n")

print(f'DT - TEST PERSON CEO MAAS : { dt_reg.predict(test_person_ceo) }') # 60000
print(f'DT - TEST PERSON MUDUR MAAS : { dt_reg.predict(test_person_mudur) }') # 12000

print("\n\n\n\n\n\n")

# -------------------------------------- RF -----------------------------------

rf_reg = RandomForestRegressor(n_estimators=20,random_state=0)
rf_reg.fit(x, y)

y_pred_rf = rf_reg.predict(x)

model = sm.OLS(y_pred_rf,x)
print(model.fit().summary()) # R-squared (uncentered):  0.724

print("\n\n\n")

print(f'RF R2 SCORE : { r2_score(y, y_pred_rf ) }') # 0.951818868839326
      
print(f'RF - TEST PERSON CEO MAAS : { rf_reg.predict(test_person_ceo) }') # 54750
print(f'RF - TEST PERSON MUDUR MAAS : { rf_reg.predict(test_person_mudur) }') # 11370

print("\n\n\n\n\n")

