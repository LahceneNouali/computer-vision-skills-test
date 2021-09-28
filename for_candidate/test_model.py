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


#Load model
model = keras.models.load_model("my_model")

#Load the image data
test_data = np.load('field.npy')
test_data = np.reshape(test_data, (200,40,60,1))

#Labels
labels = pd.read_csv('labels.csv')
label_data = pd.Series(labels['Category'])

#Normalization
lb = LabelBinarizer()
label_data = lb.fit_transform(label_data)

#predict the model
pred = model.predict(test_data, verbose=1)

index = np.argmax(pred, axis=1)
labels = lb.classes_[index]
print(labels)







