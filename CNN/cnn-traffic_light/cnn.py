#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:06:43 2024

@author: furkan
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
#Bu parametreler, veri artırma tekniklerini kontrol etmek için kullanılır. Veri artırma,eğitim verilerinin
#çeşitliliğini artırmak ve modelin daha genelleştirilmiş bir şekilde öğrenmesini sağlamak için kullanılır. 

val_data_gen = ImageDataGenerator(rescale=1./255)

train_data_flow = train_data_gen.flow_from_directory(
    'cropped_lisa_1/train_1/',
    target_size=(64, 64),
    batch_size=128,
    class_mode='categorical'
)

val_data_flow = val_data_gen.flow_from_directory(
    'cropped_lisa_1/val_1/',
    target_size=(64, 64),
    batch_size=128,
    class_mode='categorical'
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D,MaxPooling2D, Flatten

model = Sequential()

model.add(Input(shape=(64, 64, 3)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(7,activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath='traffic_lite_cnn.keras',
    monitor = 'val_loss',
    save_best_only = True,
    verbose = 1
)

model.fit(train_data_flow,
          epochs=4,
          batch_size=512,
          validation_data=(val_data_flow),
          callbacks=[checkpoint]
)

train_loss, train_accuracy = model.evaluate(train_data_flow)
val_loss, val_accuracy = model.evaluate(val_data_flow)

print('Eğitim Seti - Kayıp: {}, Doğruluk: {}'.format(train_loss, train_accuracy))
print('Doğrulama Seti - Kayıp: {}, Doğruluk: {}'.format(val_loss, val_accuracy))



