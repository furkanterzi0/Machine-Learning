# -*- coding: utf-8 -*-
"""
Created on Sun May  6 17:11:37 2024

@author: terzi
"""
import pandas as pd
from sklearn.model_selection import train_test_split
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


#XGBoost
import xgboost as xgb
model = xgb.XGBClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

'''
 XGBoost 3 onemli ozelligi
 
 yuksek verilerde iyi performans
 hızlı calisma
 problem ve modelin yorumunun mumkun olması[scale gibi durumları bypass etmek]
 
'''


