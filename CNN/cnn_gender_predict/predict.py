import cv2
import numpy as np
import matplotlib.pyplot as plt

# Eğitilmiş modelin yüklenmesi
from tensorflow.keras.models import load_model
model = load_model('cnn_model.keras')

for i in range(1, 8):
    # Fotoğrafın yüklenmesi
    image_path = f'test/{i}.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Yüz tespiti için bir Haar cascade yükleyin (OpenCV'ye önceden yüklenmiş olmalı)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Yüz tespiti
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Eğer yüz bulunamazsa, tüm resmi kullanarak tahmin yap
    if len(faces) == 0:
        resized_image = cv2.resize(image, (100, 150))  # Modelin beklentilerine uygun boyuta yeniden boyutlandır
        resized_image = resized_image / 255.0  # Resmi yeniden ölçeklendirme
        
        # Tahmin yapma
        prediction = model.predict(np.expand_dims(resized_image, axis=0))[0][0]

        # Tahmin sonucunu yazdırma
        gender = "Kadin" if prediction < 0.5 else "Erkek"
        cv2.putText(image, gender, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
                
        # Resmi gösterme
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        
    else:
        # Her bir yüz için cinsiyet tahmini yapma ve dikdörtgen çizme
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]  # Yüz bölgesini al
            face = cv2.resize(face, (100, 150))  # Modelin beklentilerine uygun boyuta yeniden boyutlandır
            face = face / 255.0  # Resmi yeniden ölçeklendirme
        
            # Tahmin yapma
            prediction = model.predict(np.expand_dims(face, axis=0))[0][0]
        
            # Yüz bölgesini kareye alma
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
            # Tahmin sonucunu yazdırma
            gender = "Kadin" if prediction < 0.5 else "Erkek"
            cv2.putText(image, gender, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
        
        # Sonucu gösterme
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
