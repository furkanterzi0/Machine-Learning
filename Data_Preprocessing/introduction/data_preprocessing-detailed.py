# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 01:24:16 2024

@author: terzi
"""
# sklearn = scikit-learn -> bilimsel kit

import pandas as pd
import numpy as np

# ------------------ ------------ -------------- veri yükleme ----------------------------------------------------#


veriler = pd.read_csv('eksikveriler.csv')
print(veriler)

# ------------------ ------------ -------------- eksik veri tamamlama [sayısal]-----------------------------------#

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
yas = veriler.iloc[:, 1:4].values  # iloc = integer location

imputer = imputer.fit(yas[:, 1:4])  # fit öğretmek için kullanılır, kolonların ort değerlerini öğrenecek
yas[:,1:4] = imputer.transform(yas[:, 1:4]) #sadece sayisal degerler icin bu yontemler kullanılır

# yas = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(yas[:,1:4])

print(yas)

'''
"mean": Eksik değerleri sütunun ortalaması ile doldurur.
"median": Eksik değerleri sütunun medyanı ile doldurur.
"most_frequent": Eksik değerleri sütunun en sık kullanılan değeri ile doldurur (mod).
"constant": Eksik değerleri belirli bir sabit değer ile doldurur. Bu durumda, fill_value parametresi kullanılır ve sabit değeri belirtmeniz gerekir.
'''

# ------------------ kategorik[nominal] veri donusumu ------ encoder ----------------------------------------------#

ulke = veriler.iloc[:,0:1].values

from sklearn import preprocessing  # -> ön isleme

labelEnCode = preprocessing.LabelEncoder()

ulke[:,0]= labelEnCode.fit_transform(veriler.iloc[:,0]) #->fit_transform yöntemi, veriyi hem eğitir 
                                                        # hem de dönüştürür bu nedenle genellikle
                                                        #eğitim verisi üzerinde kullanılır
print(ulke)

oneHotEncode = preprocessing.OneHotEncoder()
ulke= oneHotEncode.fit_transform(ulke).toarray()
# ulke = preprocessing.OneHotEncoder().fit_transform(ulke).toarray() 

print(ulke)

# ------------------ veri kümelerini birlestirme ------ concat ---------------------------------------------------#

ulkeDF = pd.DataFrame(data=ulke, index=range(22),columns=['fr','tr','us']) # dataframe-> kolon basligi ve index

yasDF = pd.DataFrame(data=yas,index=range(22),columns=['boy','kilo','yas'])

cinsiyet=veriler.iloc[:,-1].values

cinsiyetDF = pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])


s = pd.concat([ulkeDF,yasDF],axis=1) # -> axis = 1 yan yana axis=0(default) alt alta birlestirir # axis = eksen

sonuc = pd.concat([s,cinsiyetDF],axis=1) # artık veriler eksik değil ve kategorik veriler[nominal] sayısal oldu
print(sonuc)

# ------------------ verileri bölme [train and test]--------------------------------------------------------------#

from sklearn.model_selection import train_test_split

x_train, x_test, y_train , y_test = train_test_split(s,cinsiyetDF,test_size=0.33,random_state=0) 
# x bağımssız y bağımlı degisken
# test_size ->%67 train %33 test
# hedef = bağımlı degisken
# x train ve x testleri kullarak y[bağımlı degsiskeni] tahmin ettircez

# ------------------ öznitelik ölcekleme [verileri ayni dunyaya çekme]--------------------------------------------#

from sklearn.preprocessing import StandardScaler # -> verileri standartlastırmaya yarar

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
#Öznitelik ölçekleme; [verileri aynı dünyaya çevirmek birbirine yakın sayılar elde etmek]
#[birbirine göre ölçeklenmiş oldu]
# VERİLER STANDARTLAŞTIRILDI [StandartScaler]



