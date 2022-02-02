import numpy as np
import cv2 as cv
import os

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
#Path to dataset
path = 'Photos/'

data = []
target = []
List = os.listdir(path)
#Listing through the dataset and read images
for folder in List:
    for file in os.listdir(path+folder):
        img = cv.imread(os.path.join(path+folder, file))
        data.append(img/255)
        target.append(List.index(folder))
#Convert data to numpy arrays
data = np.array(data)
target = np.array(target, dtype=int)
#Simple CNN classifier model with 9 output classes
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(113, 113, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(9, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
#Fit the model through the 100 epochs, if you have large dataset you can
#use more epochs or less.
model.fit(data, target, epochs=100, batch_size=50, verbose=1)
model.save('model.h5')
