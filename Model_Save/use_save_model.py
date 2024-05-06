# -*- coding: utf-8 -*-
"""
Created on Mon May  6 19:11:57 2024

@author: terzi
"""

import pickle

with open('knn_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

import numpy as np

height = float(input("height (cm): "))
weight = float(input("weight (kg): "))
age = float(input("age: "))

data = np.array([height, weight, age]).reshape(1, -1)

result = loaded_model.predict(data)
print("result: ", result)