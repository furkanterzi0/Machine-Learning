#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 14:25:35 2024

@author: furkan
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = 'archive/Training'

data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 0,
    width_shift_range =0,
    height_shift_range= 0,
    horizontal_flip= False    
)
batch_size = 128
data_flow = data_generator.flow_from_directory(
    data_dir,
    target_size=(100, 150),
    batch_size=batch_size,
    class_mode='binary'  
)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(32, (3,3) ,activation='relu', input_shape=(100,150,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3) ,activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (3,3) ,activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    data_flow,
    epochs=3,
    batch_size=512,
    callbacks=[ModelCheckpoint('model.keras', save_best_only=True)] # En iyi modeli kaydetmek için bir geri çağrı
    )

model.save('cnn_model.keras')



