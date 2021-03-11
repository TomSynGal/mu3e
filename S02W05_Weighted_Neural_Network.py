# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:47:49 2021

@author: thoma
"""


from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import categorical_crossentropy
import keras
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import itertools
import sys

doWeightedFit = False

signal_tag = 4 #(1=2, 2=10, 3=20, 4=30, 5=40, 6=50, 7=60, 8=70)MeV

fulldata = loadtxt('mu3e_dark_photon_v00.csv', delimiter=',')

signal_dictionary = { 1:'2', 2:'10', 3:'20', 4:'30', 5:'40', 6:'50', 7:'60', 8:'70'}
lab_signal = signal_dictionary[signal_tag]+" MeV"
tag=signal_dictionary[signal_tag]+"MeV"

dataset1 = fulldata[  (fulldata[:,0] == 0)   ] # [0:5000,:]
dataset2 = fulldata[  (fulldata[:,0] == signal_tag)   ]
dataset = np.concatenate( (dataset1, dataset2), axis=0  )

X = dataset[:,2:6]
y = dataset[:,0].astype(int)//signal_tag  # to be 0 or 1
w = dataset[:,1]

print( 'Data normalization here')
scaler = StandardScaler()
scaler.fit(X)
X_norm = scaler.transform(X)

dataset_norm = np.insert(X_norm,0,y, axis=1)
dataset_norm = np.insert(dataset_norm,1,w, axis=1)

np.random.shuffle(dataset_norm)

N = dataset_norm.shape[0]

X_ = dataset_norm[:,2:6]
y_ = dataset_norm[:,0].astype(int)
w_ = dataset_norm[:,1]

N_train = N//2

X_train = X_[0:N_train,:]
y_train = y_[0:N_train]
w_train = w_[0:N_train]

X_test = X_[N_train:N,:]
y_test = y_[N_train:N]
w_test = w_[N_train:N]


if doWeightedFit:
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=4, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=150, batch_size=10, sample_weight= np.ascontiguousarray(w_train),
                        validation_split=0.5)
    print(history.history.keys())
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('weighted model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("weighted_model_accuracy.png")
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('weighted model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("weighted_model_loss.png")
    model.save("my_Weighted_model")
else:
    model = keras.models.load_model("my_Weighted_model")

predictions = model.predict(X_test, batch_size=10, verbose=0)

for i in predictions:
    print(i)

rounded_predictions = np.argmax(predictions, axis=-1)

for i in rounded_predictions:
    print(i)
    

cm = confusion_matrix(y_true = y_test, y_pred = rounded_predictions)

def plot_confusion_matrix (cm, classes,
                           normalize=False,
                           title='Confusion Matrix',
                           cmap=plt.cm.Blues):
    
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix, Without Normalization')
    
    print(cm)
    
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm_plot_labels = ['Background', 'Signal']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

    