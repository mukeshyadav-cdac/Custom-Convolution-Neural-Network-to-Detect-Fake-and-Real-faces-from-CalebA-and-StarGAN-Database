import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras

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
from keras.optimizers import Adam
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense,GlobalAveragePooling2D

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)


base_path = '/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/'

train_df = pd.read_csv("../input/140k-real-and-fake-faces/train.csv")
valid_df = pd.read_csv("../input/140k-real-and-fake-faces/valid.csv")
test_df = pd.read_csv("../input/140k-real-and-fake-faces/test.csv")

train_df.head()
df = pd.DataFrame(train_df[50010:50510])
df = df.append(train_df[0:500], ignore_index=True)

valid_df.head()
df_valid = pd.DataFrame(valid_df[10010:10060])
df_valid = df_valid.append(valid_df[0:50], ignore_index=True)

test_df.head()
df_test = pd.DataFrame(test_df[10010:10060])
df_test = df_test.append(test_df[0:50], ignore_index=True)

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

valid_X = prepareImages(df_valid, 100, "train")
valid_X /= 255

test_X = prepareImages(df_test, 100, "train")
test_X /= 255

y, label_encoder = prepare_labels(df['label'])
valid_y, valid_label_encoder = prepare_labels(df_valid['label'])
test_y, test_label_encoder = prepare_labels(df_test['label'])
base_model=MobileNet(weights='imagenet',include_top=False, input_shape=(224, 224, 3))

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(2,activation='softmax')(x) #


model = Model(inputs=base_model.input,outputs=preds)

for layer in model.layers:
    layer.trainable=False

for layer in model.layers[:25]:
    print(layer)
    layer.trainable=False
for layer in model.layers[25:]:
    print(layer)
    layer.trainable=True

model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy', metrics=[categorical_crossentropy, categorical_accuracy])

model.summary()

history = model.fit(X, y, epochs=20, batch_size=100, verbose=1, validation_data=(valid_X, valid_y))

print(history.history)

plt.figure()
plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.plot(history.history["categorical_accuracy"], label="training accuracy")
plt.plot(history.history["val_categorical_accuracy"], label="validation accuracy")
plt.title("Training/Validation Loss and Accuracy (MobileNet)")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()

print("[INFO] evaluating network...")
acc  = model.evaluate(test_X, test_y, batch_size=100)
print(acc)

from sklearn.metrics import classification_report
predictions = model.predict(x=test_X, batch_size=100)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1)))

y_val_cat_prob=model.predict(valid_X)

from sklearn.metrics import roc_curve,roc_auc_score
fpr , tpr,x  = roc_curve ( test_y.ravel() , y_val_cat_prob.ravel())

def plot_roc_curve(fpr,tpr): 
  plt.plot(fpr,tpr) 
  plt.axis([0,1,0,1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.show()    
  
plot_roc_curve (fpr,tpr) 

auc_score=roc_auc_score(test_y.ravel() , y_val_cat_prob.ravel())  #0.8822

print(auc_score)