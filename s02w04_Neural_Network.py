# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 00:55:45 2021

@author: thomas a gallagher
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

#Adjustable parameters
##############################################################################
##############################################################################

signal_tag = 4 #(1=2, 2=10, 3=20, 4=30, 5=40, 6=50, 7=60, 8=70)MeV

TrainModel = False

TrainModel_2 = False

Train_Weighted_Model = True

DoShuffle = False

DoShuffle2 = False

InputData = 1

#End of adjustable parameters
##############################################################################
##############################################################################

#GPU
##############################################################################
'''
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('Number of GPUs available ', len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
'''
##############################################################################



#Data manipulation
##############################################################################
##############################################################################
##############################################################################
#Loading the full dataset.
fulldata = loadtxt('mu3e_dark_photon_v00.csv', delimiter=',')

# shuffle the full data set
if DoShuffle:
    fulldata = np.random.shuffle(fulldata)

#Defining the energy levels within the full dataset.
signal_dictionary = { 1:'2', 2:'10', 3:'20', 4:'30', 5:'40', 6:'50', 7:'60', 8:'70'}
lab_signal = signal_dictionary[signal_tag]+" MeV"
tag=signal_dictionary[signal_tag]+"MeV"

 # Using just the data from the first column of the full data set
 # The following code combines two energy levels of data by specifying
 # Data that is the backgoround = 0 and any of the energy levels using the signal_tag
 # Concatenate joins the two pieces of the full dataset together.
dataset1 = fulldata[  (fulldata[:,0] == 0)   ] # [0:5000,:]
dataset2 = fulldata[  (fulldata[:,0] == signal_tag)   ]
dataset = np.concatenate( (dataset1, dataset2), axis=0  )

if DoShuffle2:
    dataset = np.random.shuffle(dataset)

#Finds X, y and w
#X contains meelow, meehigh, peee, Meee
#Y contains 0 for background and 1 for the signal denoted by signal_tag
#Y converts the column to integers and divides by the tag signal to show just 0 and 1
#w Contains the weights of the dataset from the first (second) column of the dataset
X = dataset[:,2:6]
y = dataset[:,0].astype(int)//signal_tag  # to be 0 or 1
w = dataset[:,1]


#Now to continue slicing the dataset

#The meelow and meeHigh value of the background in one list
X_meeLowHigh_bkg = np.concatenate( (dataset1[ :,2 ], dataset1[ :,3 ]) )
#The meelow and meeHigh value of the signal in one list
X_meeLowHigh_sig = np.concatenate( (dataset2[ :,2 ], dataset2[ :,3 ]) )
#The associated weight of the background of double length
w_double_bkg = np.concatenate( (dataset1[ :,1 ], dataset1[ :,1 ]) )
#The associated weight of the signal of double length
w_double_sig = np.concatenate( (dataset2[ :,1 ], dataset2[ :,1 ]) )

#New dataset slices (Thomas Gallagher)

##############################################################################

#The meelow of the background
X_meeLow_bkg = (dataset1[ :,2 ])
#The meelow of the signal
X_meeLow_sig = (dataset2[ :,2 ])
#The meeHigh of the background
X_meeHigh_bkg = (dataset1[ :,3 ])
#The meeHigh of the signal
X_meeHigh_sig = (dataset2[ :,3 ])
#The Peee of the background
X_Peee_bkg = (dataset1[ :,4 ])
#The Peee of the signal
X_Peee_sig = (dataset2[ :,4 ])
#The Meee of the background
X_Meee_bkg = (dataset1[ :,5 ])
#The Meee of the signal
X_Meee_sig = (dataset2[ :,5 ])
#The associated weight of the background
w_bkg = (dataset1[ :,1 ])
#The associated weight of the signal
w_sig = (dataset2[ :,1 ])

##############################################################################

#Data for model (Thomas Gallagher)

##############################################################################
'''
#test = np.concatenate(X[:,2], X[:,3])

#a = X[:,2]
#b = X[:,3]

#cofhg = np.concatenate( (a ,b) )

X_train, X_test = np.split(np.concatenate( (X[:,2], X[:,3])))

y_train, y_test = np.split(np.concatenate((y[:,2], X[:,3])))
'''
    
## calculate overall signal and background normalizations:
N_bkg_all = w[ y==0 ].sum()
N_sig_all = w[ y==1 ].sum()
print('Total bkg: ', N_bkg_all,' and signal:', N_sig_all)

print( 'Data normalization here')
scaler = StandardScaler()
scaler.fit(X)
X_norm = scaler.transform(X)

dataset_norm = np.insert(X_norm,0,y, axis=1)
dataset_norm = np.insert(dataset_norm,1,w, axis=1)
#print(dataset_norm)

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
'''
X_weighted_train0 = np.concatenate( X_train, X_train, X_train)
X_weighted_train = np.concatenate( X_weighted_train0, X_train, X_train)

X_weighted_test0 = np.concatenate( X_test, X_test, X_test)
X_weighted_test = np.concatenate( X_weighted_test0, X_test, X_test)
'''
##############################################################################

#End of data manipulation
##############################################################################
##############################################################################
##############################################################################


#Do plots and pre-model graphs
##############################################################################
##############################################################################
##############################################################################



#From the SVM helper module
##############################################################################
def makeSignalandBackgroundPlot(x_bkg, w_bkg, x_sig, w_sig, bins, xlabel, ylabel, lab_signal, title, pname, doLog=False):
    plt.clf()
    dens = True
    #Plotting a histogram for the background, the ,_,_ section somehow resizes it to match the signal
    h_main,_,_=plt.hist(x_bkg, bins=bins,weights=w_bkg,label=['background'],density=dens, color='green')
    #Plotting the second histogram for the signal background
    plt.hist(x_sig, bins=bins,weights=w_sig,label=['signal '+lab_signal], histtype=u'step', density=True, color='k')

    # plot error bars for the background
    bincenters = 0.5*(bins[1:]+bins[:-1])
    binwidths  = (bins[1:]-bins[:-1])
    Nbins = len(h_main)

    #h_int =sum( [ h_main[i]*binwidths[i] for i in range(Nbins)  ]  )
    #print('histo integral',h_int)
    #Ntot = np.sum(w_bkg)/h_int
    print(np.sum(w_bkg),(bins[-1]-bins[0]))
    Ntot = 1
    if dens:
        Ntot =  np.sum(w_bkg) *(bins[-1]-bins[0])/len(h_main)
    h_main_err = np.sqrt( np.histogram(x_bkg,bins=bins, weights=w_bkg*w_bkg)[0]  ) / Ntot
    plt.errorbar(bincenters, h_main,barsabove=True, ls='', yerr=h_main_err, marker='+',color='red')
    
    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend() 
    plt.savefig(pname+".png")
    if doLog:
        plt.yscale('log')
        plt.savefig(pname+"_log.png")
##############################################################################

##############################################################################

##### make plot
def makePlot(X1,X2, tag, Nb, **kwargs):
    plt.clf()
    
    xtitle=tag
    title = tag
    for key, value in kwargs.items():
        if key == "xtitle":
            xtitle = value
        elif key=="title":
            title = value

##############################################################################
'''
bins = np.linspace(0., 80., 40)
makeSignalandBackgroundPlot(X_meeLowHigh_bkg, w_double_bkg, X_meeLowHigh_sig, w_double_sig, bins, r'$m_{ee}^{both}$ [MeV]', 'Entries (Norm)', lab_signal, '', 'mee_both_'+tag)
'''
#End of do plots and pre-model graphs
##############################################################################
##############################################################################
##############################################################################



#Model
##############################################################################
##############################################################################
##############################################################################


#Neural network model
##############################################################################
'''
if TrainModel:
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=4, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    history = model.fit(X_weighted_train, y_train, epochs=150, batch_size=10, sample_weight = w_test,
                        validation_data=(X_weighted_test,y_test), )
  
    history = model.fit(X_train, y_train, epochs=150, batch_size=10, validation_data=(X_test,y_test)
    print(history.history.keys())
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('NN model accuracy weighted')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("model_accuracy.png")
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('NN model loss weighted')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("model_loss.png")
    model.save("my_model")
else:
    model = keras.models.load_model("my_model")
'''
##############################################################################

#Model 2
##############################################################################
'''
if TrainModel_2:
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=4, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=35, batch_size=10,
                        validation_data=(X_test,y_test))
    print(history.history.keys())
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("model_accuracy.png")
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("model_loss.png")
    model.save("my_model")
else:
    model = keras.models.load_model("my_model")
 '''  
##############################################################################

#Model 3
##############################################################################

if Train_Weighted_Model:
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=4, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=10,
                        sample_weight = w_train,
                        validation_data=(X_test,y_test, sample_weight = w_test))
    print(history.history.keys())
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("model_accuracy.png")
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("model_loss.png")
    model.save("my_weighted_model")
else:
    model = keras.models.load_model("my_weighted_model")

##############################################################################
    
    
#End of model
##############################################################################
##############################################################################
##############################################################################

#Evaluate model
##############################################################################
############################################################################## 
############################################################################## 

##############################################################################

predictions = model.predict(x=X_test, batch_size=10, verbose=0)

for i in predictions:
    print(i)

rounded_predictions = np.argmax(predictions, axis=1)

for i in rounded_predictions:
    print(i)
    
##############################################################################

#Confusion Matrix
##############################################################################

cm = confusion_matrix(y_true=y_test, y_pred=rounded_predictions)

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion Matrix',
                          cmap = plt.cm.Blues):
    
    plt.clf
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    plt.show
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalised Confusion Matrix')
    else:
        print('Confusion Matrix Without Normalisation')
    
    print(cm)
    
    thresh = cm.max() / 2
    
    for i , j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
                 horizontalalignment = 'center',
                 color='white' if cm[i,j] > thresh else 'Black')
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

cm_plot_labels = ['Background', 'Signal']
plot_confusion_matrix (cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

##############################################################################

#_, accuracy = model.evaluate(X, y)
#print('Accuracy: %.2f' % (accuracy*100))
#print(' Number of entries: ', len(X))
'''
x_bkg =  dataset_norm[  dataset_norm[:,5] == 0  ][:,2:6] 
x_sig =  dataset_norm[  dataset_norm[:,5] > 0  ][:,2:6] 

## prediction
res_sig = model.predict(x_sig)

res_bkg = model.predict(x_bkg)

#print(res_sig)
makePlot(res_bkg.flatten(),res_sig.flatten(), "nn_pred", 20, xtitle="NN output", title="all sample")

data_train = dataset_norm[N_train:,:]

x_bkg = data_train[  data_train[:,1] == 0  ][:,0:6] 
x_sig = data_train[  data_train[:,1] > 0 ][:,0:6]


## prediction
res_sig = model.predict(x_sig)
res_bkg = model.predict(x_bkg)
makePlot(res_bkg.flatten(),res_sig.flatten(), "nn_pred_test", 20, xtitle="NN output", title="test sample")



_,acc_train = model.evaluate(X_train, y_train)
print('Accuracy train: {:.2f}'.format(acc_train))

score,acc_test = model.evaluate(X_test, y_test)
print('Accuracy test: {:.2f}'.format(acc_test))

#print(score)

#print(y_test)
#print(y_test[ y_test>1  ] )

#End of evalueate model
 '''
##############################################################################
############################################################################## 
##############################################################################

#ROC and Other
##############################################################################
##############################################################################
'''
def getROC(bkg_values, sig_values, bins=None):
    
    bkg_all = bkg_values.sum()
    sig_all = bkg_values.sum()

    bkg_rej = []
    sig_eff = []
    sum_sig = 0
    sum_bkg = 0
    for j in range(len(bkg_values)):
        i =  len(bkg_values) -1 -j
        sum_sig += sig_values[i]
        sum_bkg += bkg_values[i]

        i_bkg = 1-sum_bkg/bkg_all 
        i_sig = sum_sig/sig_all
        bkg_rej.append( i_bkg )
        sig_eff.append( i_sig )

        #if i_bkg < 0.99 and i_bkg > 0.01:
        #    if i_sig < 0.99 and i_sig > 0.01:
        #        print('{:4.3f} {:4.3f}'.format(i_bkg, i_sig))
        

    return bkg_rej, sig_eff

getROC(x_bkg, x_sig)

bkg_rej, sig_eff = getROC(bkg_rej, sig_eff, bin_edges)
bkg_rej_train, sig_eff_train = getROC(bkg_values_train, sig_values_train, bin_edges_train)

bkg_rej_all, sig_eff_all = getROC(bkg_values_all, sig_values_all, bin_edges_all)


plt.clf()
plt.plot(sig_eff,bkg_rej,  '--bo', label='ROC, test')
plt.plot(sig_eff_train,bkg_rej_train, label='ROC, train')

plt.plot([0, 1], [1, 0], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Signal Efficiency')
'''
'''
plt.ylabel('Background rejection')
plt.title('ROC Curve')
plt.legend()
plt.savefig("ROC_weighted"+tag+".png")
'''

##############################################################################
##############################################################################
'''    
    
    calculate the limit 
'''   