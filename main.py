import numpy as np
import matplotlib.pyplot as plt
import keras
import pandas as pd
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras import layers
import os
from PIL import Image

'''
data = pd.read_csv("icml_face_data.csv")


def parse_data(data):
    image_array = np.zeros(shape=(len(data), 48, 48, 1))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, ' pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48, 1))
        image_array[i] = image

    return image_array, image_label


# Splitting the data into train, validation and testing set thanks to Usage column
train_imgs, train_lbls = parse_data(data[data[" Usage"] == "Training"])
val_imgs, val_lbls = parse_data(data[data[" Usage"] == "PrivateTest"])
test_imgs, test_lbls = parse_data(data[data[" Usage"] == "PublicTest"])




print(train_imgs.shape, train_lbls.shape)
print(val_imgs.shape, val_lbls.shape)
print(test_imgs.shape, test_lbls.shape)

model = keras.Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(7, activation='softmax')
])
model.summary()

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

history = model.fit(train_imgs, train_lbls, epochs=3, batch_size=32, validation_split=0.2, verbose=0)
model.save('classifier1.keras')

'''

'''
train_data = df[df[' Usage'] == 'Training']
test_data = df[df[' Usage'] == 'PublicTest']
val_data = df[df[' Usage'] == 'PrivateTest']

df[' pixels'] = df[' pixels'].apply(lambda x: string2array(x))
x_train = train_data[' pixels']

print(x_train.shape)
x_train = np.stack(x_train, axis = 0)
x_train = x_train.reshape(28709, 48, 48, 1)

y_train = to_categorical(train_data['emotion'])

x_test = test_data[' pixels']
x_test = np.stack(x_test, axis = 0)
x_test = x_test.reshape(3589 , 48, 48, 1)

y_test = to_categorical(test_data['emotion'])

x_val = val_data[' pixels']
x_val = np.stack(x_val, axis = 0)
x_val = x_val.reshape(3589 , 48, 48, 1)

y_val = to_categorical(val_data['emotion'])

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_val.shape, y_val.shape)
'''

'''
model = Sequential()
model.add(Conv2D(48, (3, 3), activation="relu", input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(96, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(96, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(96, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=64, epochs=25, validation_data=(x_test, y_test))
model.save('classifier.keras')

'''

# (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
model = keras.models.load_model('classifier1.keras')

i = Image.open("WIN_20240324_13_55_02_Pro.jpg").convert('L')

i = i.resize((48, 48), resample=Image.NEAREST)
imagedata = np.asarray(i)
imagedata = imagedata.reshape((1, 48, 48, 1))
prediction = model.predict(imagedata)
print(classes[np.argmax(prediction)])

print(prediction)