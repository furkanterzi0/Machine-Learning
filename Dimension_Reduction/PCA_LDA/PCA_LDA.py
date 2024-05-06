# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:20:38 2024

@author: terzi
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


data = pd.read_csv('Wine.csv')

X = data.iloc[:,0:-1].values
y= data.iloc[:,-1].values

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.33,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train,y_train) # supervised olduğu icin y degeri de verilmeli
X_test_lda = lda.transform(X_test)


# classifier
lg = LogisticRegression(random_state=42)
lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)

lg2 = LogisticRegression(random_state=42)
lg2.fit(X_train_pca, y_train)
y_pred_pca = lg2.predict(X_test_pca)

lg3 = LogisticRegression(random_state=42)
lg3.fit(X_train_lda, y_train)
y_pred_lda = lg3.predict(X_test_lda)

#confusion matrix
print(confusion_matrix(y_test, y_pred))
print("\nPCA:\n " , confusion_matrix(y_test, y_pred_pca))
print("\nLDA:\n " , confusion_matrix(y_test, y_pred_lda))


