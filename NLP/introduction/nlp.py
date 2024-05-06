# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:42:44 2024

@author: terzi
"""

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('Restaurant_Reviews.csv', sep='\,\s*(?=[^,]*$)', engine='python')
# sep parametresi default olarak ","dır (sutunlar arası ayracı belirlemek)
# sep='\,\s*(?=[^,]*$)' -> son virgülden sonrasını ayırmak için

import re # regular expression
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # kokleri ayırmak için loved -> love
ps = PorterStemmer()

reviews=[]

y=[]
for i in range(1000):
    if not (data['Liked'][i] == "0" or data['Liked'][i] == "1"):
        continue
    review = re.sub('[^a-zA-Z]',' ',data['Review'][i])
    #Python'da ^ karakteri, bir ifadenin başında kullanıldığında mantıksal bağlamda "değil" anlamına da gelir.
    #^abc: Bu desen, metnin başlangıcında "abc" karakter dizisini arar.
    #^[\d]: Bu desen, metnin başlangıcında herhangi bir rakamı arar.
    #^[A-Z]: Bu desen, metnin başlangıcında herhangi bir büyük harfi arar.
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    reviews.append(review)
    
    if data['Liked'][i] == "1" or data['Liked'][i] == "0":
        y.append(data['Liked'][i])
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
x = cv.fit_transform(reviews).toarray()


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)




