# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:19:32 2024

@author: terzi
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

data = pd.read_csv('Churn_Modelling.csv')

X = data.iloc[:,3:-1]

le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])

ohe = OneHotEncoder()
geography_encoded = pd.DataFrame(data=ohe.fit_transform(X[['Geography']]).toarray(), index=range(len(X)),columns=["France","Spian","Germany"])

X = pd.concat([geography_encoded,X],axis=1)
X.drop(columns=['Geography'],inplace=True)

y = data.iloc[:,-1]

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.33,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Artificial Neural Network

from keras.models import Sequential # yapay sinir agi oluşturmak icin
from keras.layers import Dense # noron yapısını kurabilmek icin layer=katman

model = Sequential()

model.add(Dense(units=6, activation='relu', input_dim=X_train.shape[1])) #units = gizli katman 
model.add(Dense(units=6, activation='relu')) # one more hidden layer
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

y_pred = model.predict(X_train)

