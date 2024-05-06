# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 21:18:00 2024

@author: terzi
"""
import pandas as pd 
data = pd.read_csv('emotion_data.csv')

# preprocessing
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words ]
    return ' '.join(tokens)

data['clean_text'] = data['Text'].apply(preprocess_text)
# text kolonuna fonksiyonu uygula

#TF-IDF -ozellik cikarimi-
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_text'])

#train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, data['Emotion'], test_size=0.2, random_state=42)

# machine learning -svc-
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

# user input
sentence = str(input('sentence : '))
clean_sentence = preprocess_text(sentence)
print("clean sentence: ",clean_sentence)
clean_vectorized_sentence = vectorizer.transform([clean_sentence])

result = model.predict(clean_vectorized_sentence)
print("emotion: ",result)

