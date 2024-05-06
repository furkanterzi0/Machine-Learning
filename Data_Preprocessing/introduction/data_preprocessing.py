# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 02:00:56 2024

@author: terzi
"""
#data = pd.DataFrame({'category': ['A', 'B', 'A', 'C']})

#1 veri okuma 
#2 ekik sayısal veriyi ortalama ile tamamlama
#3 sözel ifadeleri kategorik(nominal) sayıya çevirme [one hot encode]
#4 verileri birlestirme
#5 test ve eğitim verilerini ayırma
#6 öznitelik ölcekleme 

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer # impute = doldurmak veya tamamlamak denebilir
from sklearn.preprocessing import OneHotEncoder , LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#1
data = pd.read_csv('eksikveriler.csv')

#2
age = data.iloc[:,3:4].values  ## buradda iloc ile ,3:4 kullanma sebebi iki boyutlu dizi elde etmek
age = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(age)

data['yas'] = age

#3
country = OneHotEncoder().fit_transform(data.iloc[:,0:1]).toarray()

#OneHotEncoder().fit_transform(data.iloc[:,4:5].toarray()) # cinsiyeti donusturme

#4
countryDF = pd.DataFrame(data=country, index=range(22), columns=['fr','tr','us'])
dataDF = pd.DataFrame(data=data.iloc[:,0:4], index=range(22), columns=['boy','kilo','yas']) #x->bağımsız veri
genderDF = pd.DataFrame(data=data.iloc[:,-1].values, index=range(22), columns=['gender']) #y->hedef->bagımlı veri

dataDF = pd.concat([countryDF,dataDF],axis=1)

#5
x_train, x_test, y_train, y_test = train_test_split(dataDF, genderDF, test_size=0.33, random_state=0)

#6
X_train = StandardScaler().fit_transform(x_train)
X_test = StandardScaler().fit_transform(x_test)

# [bağımsız veriler standart scaler ile birbirine göre ölceklenmis oldu]
print(f'{dataDF} \n\n {genderDF}')


