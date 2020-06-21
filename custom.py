import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D,AveragePooling2D
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

print(tf.__version__)

import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

base_path = '/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/'

train_df = pd.read_csv("../input/140k-real-and-fake-faces/test.csv")
valid_df = pd.read_csv("../input/140k-real-and-fake-faces/valid.csv")
test_df = pd.read_csv("../input/140k-real-and-fake-faces/test.csv")


train_df.head()
df = pd.DataFrame(train_df[15010:15510])
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
        img = image.load_img( base_path + fig, target_size=(224, 224, 3))
        x = image.img_to_array( img)
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

model = Sequential()

model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (224, 224, 3)))

model.add(BatchNormalization(axis = 3, name = 'bn0'))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool'))
model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
model.add(Activation('relu'))
model.add(AveragePooling2D((3, 3), name='avg_pool'))

model.add(Flatten())
model.add(Dense(500, activation="relu", name='rl'))
model.add(Dropout(0.8))
model.add(Dense(2, activation='softmax', name='sm'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

history = model.fit(X, y, epochs=20, batch_size=100, verbose=1, validation_data=(valid_X, valid_y))

print(history.history)

plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.plot(history.history["accuracy"], label="training accuracy")
plt.plot(history.history["val_accuracy"], label="validation accuracy")
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
