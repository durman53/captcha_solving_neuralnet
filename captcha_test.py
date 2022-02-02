import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model

List = ['airplane', 'bicycle', 'boat', 'car',
        'motorbus', 'motorcycle', 'seaplane',
        'train', 'truck']
#Getting image path from user
img_path = input('image path: ')
#Read image
data = []
img = cv.imread(img_path)
img = cv.resize(img, (113, 113))
data.append(img/255)
data = np.array(data)
#Load pretrained model
model = load_model('model.h5')
#Predict and show results
pred = model.predict(data)
print(f'Predicted class for image is {pred[0][np.argmax(pred[0])]*100:.2f}% {List[np.argmax(pred[0])]}')
cv.imshow('image', img)
