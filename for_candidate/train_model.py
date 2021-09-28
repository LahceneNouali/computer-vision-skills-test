import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

import keras 
import keras.backend as k
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers import Dense,Dropout,Flatten
from keras import losses
from tensorflow.keras.optimizers import Adam,RMSprop,Adadelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.applications.vgg16 import preprocess_input,decode_predictions
import matplotlib.pyplot as plt


num_classes = 2

#Load the image data
class_a = np.load('class_a.npy')
class_b = np.load('class_b.npy')

#Concatenate 2 numpy arrays
train_data = np.concatenate((class_a, class_b))
train_data = np.reshape(train_data, (2000,40,60,1))

#Train labels
labels = pd.read_csv('labels.csv')
label_data = pd.Series(labels['Category'])

#Normalization
lb = LabelBinarizer()
label_data = lb.fit_transform(label_data)

#Split the train dataset
X_train,X_val,Y_train,Y_val = train_test_split(train_data, label_data, test_size=0.2, random_state=431)

#CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(40,60,1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#Compile the model
model.compile(loss=losses.sparse_categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])

cnn = model.fit(X_train, Y_train, batch_size=128, epochs=50, verbose=1, validation_data=(X_val,Y_val), shuffle=True)

#Save model
model.save('my_model')

#Plots fro training and validation process: loss and accuracy
plt.figure(figsize=(10,6))
plt.plot(cnn.history['accuracy'], 'g')
plt.plot(cnn.history['val_accuracy'], 'b')
plt.xticks(np.arange(1,60,2))
plt.title('training accuracy vs validation accuracy')
plt.xlabel('No. of Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'validation'])
plt.show()

plt.figure(figsize=(10,6))
plt.plot(cnn.history['loss'], 'g')
plt.plot(cnn.history['val_loss'], 'b')
plt.xticks(np.arange(1,60,2))
plt.title('training loss vs validation loss')
plt.xlabel('No. of Epochs')
plt.ylabel('loss')
plt.legend(['train', 'validation'])
plt.show()
