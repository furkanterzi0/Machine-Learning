# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:38:16 2024

@author: terzi
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


original_data= pd.read_csv('veriler.csv')

country = original_data.iloc[:,:1].values
gender = original_data.iloc[:,4:5].values

data= original_data.iloc[:,1:4]

countryOHE = OneHotEncoder().fit_transform(country).toarray()
countryDF = pd.DataFrame(data = countryOHE, index=range(22), columns=['fr','tr','us'] )

genderOHE = OneHotEncoder().fit_transform(gender).toarray()
genderDF = pd.DataFrame(data = genderOHE, index=range(22), columns=['E','K'] )

data = pd.concat([countryDF,data],axis=1)


                                                    
x_train, x_test, y_train, y_test = train_test_split(data,genderDF.iloc[:,0:1],test_size=0.33, random_state=1)
                                                        # cinsiyet kolonundan sadece birini aldık
                                                        # cunku erkek değilse kadındır [dummy veriable]

regressor = LinearRegression().fit(x_train,y_train)

y_pred = regressor.predict(x_test)   # cinsiyet tahmini


# Geri Eleme (Backward Elimination)

import numpy as np
import statsmodels.api as sm # stat-> istatistik

X = np.append(arr=np.ones((22,1)).astype(int),  values=data, axis=1) # axis = 1 -> kolon olarak eklemek
# burada birlerden oluşan bir kolon ekliyoruz, denklemdeki çarpan değeri 1[etkisiz eleman]
#  y=b0 +b1x1 +b2x2.......+e.     aslında b0'i ekliyoruz

X_list = data.iloc[:,[0,1,2,3,4,5]].values # list olarak tanımlıyoruz ki
# analiz yapıldıgında hangisinin p degeri yuksek görelim ve onu çikaralim
X_list = np.array(X_list,dtype=float) # np arraya dönüstürmek

model = sm.OLS(genderDF.iloc[:,0:1],X_list).fit() # 0:1 erkek değilse kadındır dummy veraible
print(model.summary())
# bazı p degerleri 0.05'in üstünde çıktı öncelikle en buyuk cıkanı çıkartıp tekrar deneme yanılma 
# yöntemiyle devam edilebilir, p degerleri dustukce sonuca yaklasiyo sayilabiliriz ;

# X_list = data.iloc[:,[3,4,]].values gibi





