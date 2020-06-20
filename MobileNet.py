import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras import layers
from keras.preprocessing import image
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

base_path = '/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/'

train_df = pd.read_csv("../input/140k-real-and-fake-faces/test.csv")
valid_df = pd.read_csv("../input/140k-real-and-fake-faces/valid.csv")


train_df.head()
df = pd.DataFrame(train_df[10010:10510])
df = df.append(train_df[0:500], ignore_index=True)
df

valid_df.head()
df_z = pd.DataFrame(valid_df[10010:10510])
df_z = df_z.append(valid_df[0:500], ignore_index=True)
df_z

def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 224, 224, 3))
    count = 0
    
    for fig in data['path']:
        #load images into images of size 100x100x3
        img = image.load_img( base_path + fig, target_size=(224, 224, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        
        print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train


def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder

X = prepareImages(df, 1000, "train")
X /= 255

valid_X = prepareImages(df_z, 1000, "train")
valid_X /= 255

y, label_encoder = prepare_labels(df['label'])
valid_y, valid_label_encoder = prepare_labels(df_z['label'])

def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

model = MobileNet(input_shape=(224, 224, 3), alpha=1., weights=None, classes=2)
model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
print(model.summary())


history = model.fit(X, y, epochs=5, batch_size=100, verbose=1, validation_data=(valid_X, valid_y))

plt.plot(history.history['categorical_accuracy'])
plt.title('Model categorical accuracy')
plt.ylabel('categorical accuracy')
plt.xlabel('Epoch')
plt.show()