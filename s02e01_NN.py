# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:56:42 2021

@author: thoma
"""
##############################################################################
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import categorical_crossentropy
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import itertools
import sys
import scipy.stats
from math import sqrt, isinf
from scipy.signal import savgol_filter
from Functions import SignalBackgroundPlot, CalculateHistogramLimits, MakePlot
from Functions import plot_confusion_matrix, MakeLogPlot
##############################################################################
#
#
#
##############################################################################
#Welcome to the mu3e Code suite!
#Please download and use Functions.py to run this code.
#Please select which analysis methods are to be performed.
#01/12 Analysis Selection
##############################################################################

DoCutBasedAnalysis = False

DoDataScaling = True

DoWeightScaling = True

DoNeuralNetwork = False

ReTrainNeuralNetwork = False

DoAnalysis = False

DoWeightedNeuralNetwork = True

ReTrainWeightedNeuralNetwork = False

DoWeightedAnalysis = True

DoSVM = False

##############################################################################
#End 01/12 Analysis Selection
##############################################################################
#
#
#
##############################################################################
#Please select the signal Energy
#(1=2, 2=10, 3=20, 4=30, 5=40, 6=50, 7=60, 8=70)MeV
#02/12 Energy Selection
##############################################################################

signal_tag = tag_signal = 7

##############################################################################
#End of 02/12 Energy Selection
##############################################################################
#
#
#
##############################################################################
#03/12 Data Load and Preprocessing
##############################################################################

print('Loading data...')

fulldata = loadtxt('mu3e_dark_photon_v01.csv', delimiter=',')

signal_dictionary = dict_signals = { 1:'2', 2:'10', 3:'20', 4:'30', 5:'40', 6:'50', 7:'60', 8:'70'}
lab_signal = signal_dictionary[signal_tag]+" MeV"
tag=signal_dictionary[signal_tag]+"MeV"
label = 'IC vs Dark Photon '+dict_signals[tag_signal]+' MeV'

#Seperating the desired signal samples and joining it to the bacjground samples.
dataset1 = fulldata[  (fulldata[:,0] == 0)   ] #Background Samples
dataset2 = fulldata[  (fulldata[:,0] == signal_tag)   ] #Signal Samples


if DoWeightScaling:
    
    print('Will scale the weights for the neural networks.')
    
    BackgroundWeights = dataset1[:,1]
    MeanBackgroundWeight = np.mean(BackgroundWeights)
    MeanAveragedBackgroundWeights = BackgroundWeights/MeanBackgroundWeight
    dataset11 = np.insert(dataset1, 1, MeanAveragedBackgroundWeights, axis = 1)
    dataset1 = np.delete(dataset11, 2, axis=1)
 
    SignalWeights = dataset2[:,1]
    MeanSignalWeight = np.mean(SignalWeights)
    MeanAveragedSignalWeights = SignalWeights/MeanSignalWeight
    dataset22 = np.insert(dataset2, 1, MeanAveragedSignalWeights, axis = 1)
    dataset2 = np.delete(dataset22, 2, axis=1)
    
else:
    
    print('Weight scaling deselected. No weight scaling will be performed.')
    
    
dataset = np.concatenate( (dataset1, dataset2), axis=0  ) #Combination Data

#Finding tags, weights and sata from the sample columns.
X = dataset[:,2:8]
y = dataset[:,0].astype(int)//signal_tag  # to be 0 or 1
w = dataset[:,1]

if DoDataScaling:
    
    print('Will scale the data for the neural networks.')
    
    scaler = StandardScaler()
    scaler.fit(X)
    X_norm = scaler.transform(X)
    
else:
    
    print('Data scaling deselected. No scaling or normalisation will be performed')
    
    X_norm = X

dataset_norm = np.insert(X_norm,0,y, axis=1)
dataset_norm = np.insert(dataset_norm,1,w, axis=1)

np.random.shuffle(dataset_norm)

N = dataset_norm.shape[0]

X_ = dataset_norm[:,2:8]
y_ = dataset_norm[:,0].astype(int)
w_ = dataset_norm[:,1]

#Train 75% of the available data.
N_train = int(3*N/4)

X_train = X_[0:N_train,:]
y_train = y_[0:N_train]
w_train = w_[0:N_train]

X_test = X_[N_train:N,:]
y_test = y_[N_train:N]
w_test = w_[N_train:N]

dataset_norm_bkg = dataset_norm[ dataset_norm[:,0]==0]
dataset_norm_sig = dataset_norm[ dataset_norm[:,0]==1]

x_bkg = dataset_norm_bkg[:,2:8]
x_sig = dataset_norm_sig[:,2:8]

trainDataAndLabels = np.insert(X_train,0,y_train, axis=1)

trainData_bkg = trainDataAndLabels[ trainDataAndLabels[:,0]==0]
trainData_sig = trainDataAndLabels[ trainDataAndLabels[:,0]==1]

x_train_bkg = trainData_bkg[:,1:7]
x_train_sig = trainData_sig[:,1:7]

testDataAndLabels = np.insert(X_test,0,y_test, axis=1)

testData_bkg = testDataAndLabels[ testDataAndLabels[:,0]==0]
testData_sig = testDataAndLabels[ testDataAndLabels[:,0]==1]

x_test_bkg = testData_bkg[:,1:7]
x_test_sig = testData_sig[:,1:7]

#New Shit
x_bkg_weight = dataset_norm_bkg[:,1]
x_sig_weight = dataset_norm_sig[:,1]

print('Data loading and pre-processing successful.')

##############################################################################
#End 03/12 Data Load and Preprocessing
##############################################################################
#
#
#
##############################################################################
#04/12 Cut-Based Analysis
##############################################################################

if DoCutBasedAnalysis:
    
    print('Will perform a cut-based analysis of the '+dict_signals[tag_signal]+' MeV signal.')
    
    X_mass_bkg = np.concatenate( (dataset1[ :,2 ], dataset1[ :,3 ]) )
    X_mass_sig = np.concatenate( (dataset2[ :,2 ], dataset2[ :,3 ]) )
    w_mass_bkg = np.concatenate( (dataset1[ :,1 ], dataset1[ :,1 ]) )
    w_mass_sig = np.concatenate( (dataset2[ :,1 ], dataset2[ :,1 ]) )

    bins = np.linspace(0., 80., 40)
    SignalBackgroundPlot(X_mass_bkg, w_mass_bkg, X_mass_sig, w_mass_sig, bins, r'$m_{ee}^{both}$ [MeV]', 'Normalised Signal Count', lab_signal, '', 'mee_both_'+tag)
    
    
    standard_limit = CalculateHistogramLimits(X_mass_bkg, w_mass_bkg, X_mass_sig, w_mass_sig, bins)

    print('The standard limit is:', standard_limit)
    
    ## calculate overall signal and background normalizations:
    N_bkg_all = w[ y==0 ].sum()
    N_sig_all = w[ y==1 ].sum()
    print('Total bacground: ', N_bkg_all,' Total signal:', N_sig_all)
    
    print('Cut-based analysis of the '+dict_signals[tag_signal]+' MeV signal successful.')
    
else:
    
    print('Cut-based analysis deselected. No anaylasis of this type will be performed.')

##############################################################################
#End 04/12 Cut-Based Analysis
##############################################################################
#
#
#
##############################################################################
#05/12 Neural Network Without Weights
##############################################################################

if DoNeuralNetwork:
    
    print('Neural network selected, preprocessing...')
    
    if ReTrainNeuralNetwork:


        model = Sequential()
        model.add(Dense(8, input_dim=6, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=300, batch_size=5,
                            validation_split=0.2)
        print(history.history.keys())
        plt.clf()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig("model_accuracy.png")
        plt.clf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig("model_loss.png")
        model.save("my_model")
    else:
        model = keras.models.load_model("my_model")

else:
    print('Neural network deselected. No analysis of this type will be performed')
    
##############################################################################
#End 05/12 Neural Network Without Weights
##############################################################################
#
#
#
##############################################################################
#05/12 Neural Network With Weights Applied
##############################################################################

if DoWeightedNeuralNetwork:
    
    print('Weighted neural network selected, preprocessing...')
    
    if ReTrainWeightedNeuralNetwork:


        modelw = Sequential()
        modelw.add(Dense(8, input_dim=6, activation='relu'))
        modelw.add(Dense(8, activation='relu'))
        modelw.add(Dense(12, activation='relu'))
        modelw.add(Dense(1, activation='sigmoid'))
        modelw.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = modelw.fit(X_train, y_train, epochs=250, batch_size=10, sample_weight= w_train,
                            validation_split=0.2)
        print(history.history.keys())
        plt.clf()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Weighted Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig("weighted_model_accuracy.png")
        plt.clf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Weighted Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig("weighted_model_loss.png")
        modelw.save("my_weighted_model")
    else:
        modelw = keras.models.load_model("my_weighted_model")

else:
    
    print('Weighted neural network deselected. No analysis of this type will be performed')
    
##############################################################################
#End 05/12 Neural Network With Weights Applied
##############################################################################
#
#
#
##############################################################################
#06/12 Neural Network Analysis
##############################################################################

if DoAnalysis:
    
    print('Will use the neural network to analyse the '+dict_signals[tag_signal]+' MeV signal.')
    
    #Predictions
    res_sig = model.predict(x_sig)
    res_bkg = model.predict(x_bkg)
    MakePlot(res_bkg.flatten(),res_sig.flatten(), "nn_pred", 20, xtitle="Neural Network Output", title="Weighted Neural Network Output For All Samples")
    MakeLogPlot(res_bkg.flatten(),res_sig.flatten(), "nn_pred_log", 20, xtitle="Neural Network Output", title="Logarithmic Neural Network Output For All Samples")
    
    res_sig = model.predict(x_train_sig)
    res_bkg = model.predict(x_train_bkg)
    MakePlot(res_bkg.flatten(),res_sig.flatten(), "nn_pred_train", 20, xtitle="Neural Network Output", title="Weighted Neural Network Output For Train Samples")
    MakeLogPlot(res_bkg.flatten(),res_sig.flatten(), "nn_pred_train_log", 20, xtitle="Neural Network Output", title="Logarithmic Neural Network Output For Train Samples")
    
    res_sig = model.predict(x_test_sig)
    res_bkg = model.predict(x_test_bkg)
    MakePlot(res_bkg.flatten(),res_sig.flatten(), "nn_pred_test", 20, xtitle="Neural Network Output", title="Weighted Neural Network Output For Test Samples")
    MakeLogPlot(res_bkg.flatten(),res_sig.flatten(), "nn_pred_test_log", 20, xtitle="Neural Network Output", title="Logarithmic Neural Network Output For Test Samples")
    
else:
    
    print('Weighted neural network analysis deselected. No analysis of this type will be performed')

##############################################################################
#End 06/12 Neural Network Analysis
##############################################################################
#
#
#
##############################################################################
#07/12 Weighted Neural Network Analysis
##############################################################################

if DoWeightedAnalysis:
    
    print('Will use the weighted neural network to analyse the '+dict_signals[tag_signal]+' MeV signal.')
    
    #Predictions
    res_sig = modelw.predict(x_sig)
    res_bkg = modelw.predict(x_bkg)
    MakePlot(res_bkg.flatten(),res_sig.flatten(), "Weighted_nn_pred", 20, xtitle="Neural Network Output", title="Weighted Neural Network Output For All Samples")
    MakeLogPlot(res_bkg.flatten(),res_sig.flatten(), "Weighted_nn_pred_log", 20, xtitle="Neural Network Output", title="Logarithmic Weighted Neural Network Output For All Samples")
    
    res_sig = modelw.predict(x_train_sig)
    res_bkg = modelw.predict(x_train_bkg)
    MakePlot(res_bkg.flatten(),res_sig.flatten(), "Weighted_nn_pred_train", 20, xtitle="Neural Network Output", title="Weighted Neural Network Output For Train Samples")
    MakeLogPlot(res_bkg.flatten(),res_sig.flatten(), "Weighted_nn_pred_train_log", 20, xtitle="Neural Network Output", title="Logarithmic Weighted Neural Network Output For Train Samples")
    
    res_sig = modelw.predict(x_test_sig)
    res_bkg = modelw.predict(x_test_bkg)
    MakePlot(res_bkg.flatten(),res_sig.flatten(), "Weighted_nn_pred_test", 20, xtitle="Neural Network Output", title="Weighted Neural Network Output For Test Samples")
    MakeLogPlot(res_bkg.flatten(),res_sig.flatten(), "Weighted_nn_pred_test_log", 20, xtitle="Neural Network Output", title="Logarithmic Weighted Neural Network Output For Test Samples")
    
    
    signalpredictions = modelw.predict(x_sig)
    
    for i in signalpredictions:
        print(i)
    
    rounded_signalpredictions = modelw.predict_classes(x_sig)
    
    for i in rounded_signalpredictions:
        print(i)
        
    rounded_predictions = modelw.predict_classes(X_test)       
    cm = confusion_matrix(y_true = y_test, y_pred = rounded_predictions) 
    cm_plot_labels = ['Background', 'Signal']
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, confusionName = 'weighted_confusion', title='Confusion Matrix')

else:
    
    print('Weighted neural network analysis deselected. No analysis of this type will be performed')

##############################################################################
#End 07/12 Weighted Neural Network Analysis
##############################################################################
#
#
#
##############################################################################
#08/12 Appendix Code
##############################################################################

    '''
    r_probs = [0 for _ in range(len(y_test))]
    rf_probs = modelw.predict_proba(X_test)
    nb_probs = modelw.predict_proba(X_train)
    
    
    rf_probs = rf_probs[:, 1]
    nb_probs = nb_probs[:, 1]
    
    r_auc = roc_auc_score(y_test, r_probs)
    rf_auc = roc_auc_score(y_test, rf_probs)
    nb_auc = roc_auc_score(y_test, nb_probs)
    
    print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
    print('Random Forest: AUROC = %.3f' % (rf_auc))
    print('Naive Bayes: AUROC = %.3f' % (nb_auc))
    
    r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)
    
    plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
    plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)
    plt.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes (AUROC = %0.3f)' % nb_auc)

    # Title
    plt.title('ROC Plot')
    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Show legend
    plt.legend() # 
    # Show plot
    plt.show()
    
    '''
    
    '''
    #testing new analysis
    
    prob_test_background = modelw.predict(x_test_bkg)
    prob_test_signal = modelw.predict(x_test_sig)
    prob_train_background = modelw.predict(x_train_bkg)
    prob_train_signal = modelw.predict(x_test_sig)
    
    prob_bkg_all = np.concatenate((prob_train_background,prob_test_background))
    weight_bkg_all = np.concatenate( (w_train_background, w_test_background))
    prob_sig_all = np.concatenate((prob_train_signal,prob_test_signal))
    weight_sig_all = np.concatenate( (w_train_signal, w_test_signal))

    limit = calcLimitFromHisto(prob_bkg_all, weight_bkg_all, prob_sig_all, weight_sig_all, bins1)
    
    bkg_values_all, bin_edges_all = np.histogram(prob_bkg_all, bins=bins1, weights=weight_bkg_all, density=True)
    sig_values_all, bin_edges_all = np.histogram(prob_sig_all, bins=bins1, weights=weight_sig_all, density=True)

    print(r'the limit is: $\times$ {:.2f} '.format( limit) )


    makeSignalBackPlot(prob_bkg_all, weight_bkg_all, prob_sig_all, weight_sig_all, bins, 'Probability of signal', 'Entries (Norm.)', lab_signal, label+' weighted frames, test+train sample', "prob_distribution_"+tag+"_all",True)
    
    ## roc curves
    bkg_rej, sig_eff = getROC(bkg_values, sig_values, bin_edges)
    bkg_rej_train, sig_eff_train = getROC(bkg_values_train, sig_values_train, bin_edges_train)

    bkg_rej_all, sig_eff_all = getROC(bkg_values_all, sig_values_all, bin_edges_all)


    plt.clf()
    plt.plot(sig_eff,bkg_rej,  '--bo', label='ROC, test')
    plt.plot(sig_eff_train,bkg_rej_train, label='ROC, train')

    plt.plot([0, 1], [1, 0], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background rejection')
    plt.title('ROC: '+label)
    plt.legend()
    plt.savefig("ROC_weighted"+tag+".png")
   
   
    prob_train = modelw.predict(X_train)
    prob_test = modelw.predict(X_test)
    
    '''



