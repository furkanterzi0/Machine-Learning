#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:24:22 2024

@author: furkan
"""

from tensorflow.keras.models import load_model

model = load_model('traffic_light_cnn.keras')

#test_folder_dir = 'cropped_lisa_1/val_1/stop'
test_folder_dir = 'test/'

from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(image_path):
    
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    
    return img_array

import os

for img in os.listdir(test_folder_dir):
    
    img_name =str(os.path.join(test_folder_dir, img))
    
    if img_name.endswith(('.png','jpg','jpeg')):
        
        pred = model.predict(preprocess_image(img_name))
        
        class_name=['go','goForward','goLeft','stop','stopLeft','warning','warningLeft']
        result =''
        
        for i in range (len(class_name)):
            
            if pred[0][i] == 1:
                result = class_name[i]
                
        import matplotlib.pyplot as plt
        
        plt.imshow(plt.imread(img_name))
        plt.title(result, color='blue')  
        plt.axis('off')
        plt.show()
        
        print(result)
        
    