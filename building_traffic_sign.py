import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score


data = []
labels = []
cur_path = os.getcwd()

for classe in range(43):
    path = os.path.join(cur_path,'train',str(classe))
    images = os.listdir(path)

    for pic in images:
        try:
            picture = Image.open(path + '\\'+ pic)
            picture = picture.resize((30,30))
            picture = np.array(picture)
            data.append(picture)
            labels.append(classe)
        except:
            print("Error")
            
labels = np.array(labels)
data = np.array(data)


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.8, random_state=123)


y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


mmdl = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))
model.save("my_model.h5")

X_test=np.array(data)
y_test = pd.read_csv('Test.csv')

labels_test = y_test["ClassId"].values
images_test = y_test["Path"].values

d=[]

for img in images_test:
    picture = Image.open(img)
    picture = picture.resize((30,30))
    d.append(np.array(picture))


pred = model.predict_classes(X_test)
print(accuracy_score(labels_test, pred))
