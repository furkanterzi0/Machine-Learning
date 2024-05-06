# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 18:08:15 2024

@author: terzi
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

data = pd.read_csv('diabetes.csv')

x = data.iloc[:,:-1].values
y = data.iloc[:,-1:].values
y = np.ravel(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=1)

sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

rfc = RandomForestClassifier(n_estimators=50,criterion='entropy')
rfc.fit(x_train_sc,y_train)

y_pred = rfc.predict(x_test_sc)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print("accuracy score : ",accuracy_score(y_test, y_pred))

y_probs = rfc.predict_proba(x_test_sc) #proba -> muhtemelen      olasılıklar 

y_probs = y_probs[:, 1]


fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# ROC eğrisini çiz
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()