##
## written and tested with python3
##
from numpy import loadtxt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from joblib import dump, load
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,precision_recall_curve,roc_curve

from helper import getROC, getLimit,  makeSignalBackPlot, calcLimitFromHisto

import sys

'''In original code, doTrain=False and tag_signal=4 were commented out'''

doTrain = True
tag_signal = 4
dict_signals = { 1:'2', 2:'10', 3:'20', 4:'30', 5:'40', 6:'50', 7:'60', 8:'70'}

def main(doTrain=False, tag_signal=4):
    label = 'IC vs Dark photon '+dict_signals[tag_signal]+' MeV'
    lab_signal = dict_signals[tag_signal]+" MeV"
    tag=dict_signals[tag_signal]+"MeV"
    svm_file = 'mytestSVM_'+tag+'.joblib'

    print(label)
    
    dataset = loadtxt('mu3e_dark_photon_v00.csv', delimiter=',')


    ## select only 0 and 4 (30 MeV)
    dataset1 = dataset[  (dataset[:,0] == 0)   ] # [0:5000,:]
    dataset2 = dataset[  (dataset[:,0] == tag_signal)   ]
    dataset = np.concatenate( (dataset1, dataset2), axis=0  )


    X = dataset[:,2:6]
    y = dataset[:,0].astype(int)//tag_signal  # to be 0 or 1
    w = dataset[:,1]

    #### plot the standard extraction of limit
    ##########################################
    X_mass_bkg = np.concatenate( (dataset1[ :,2 ], dataset1[ :,3 ]) )
    X_mass_sig = np.concatenate( (dataset2[ :,2 ], dataset2[ :,3 ]) )
    w_mass_bkg = np.concatenate( (dataset1[ :,1 ], dataset1[ :,1 ]) )
    w_mass_sig = np.concatenate( (dataset2[ :,1 ], dataset2[ :,1 ]) )

    bins = np.linspace(0., 80., 40)
    makeSignalBackPlot(X_mass_bkg, w_mass_bkg, X_mass_sig, w_mass_sig, bins, r'$m_{ee}^{both}$ [MeV]', 'Entries (Norm)', lab_signal, '', 'mee_both_'+tag)

    standard_limit = calcLimitFromHisto(X_mass_bkg, w_mass_bkg, X_mass_sig, w_mass_sig, bins)

    print('Standard limit is:', standard_limit)
    
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


    #print(y_train)
    #sys.exit()

    clf = None
    if doTrain:
        print("Will train the support vector machine")
        clf = svm.SVC(kernel='rbf', probability=True, cache_size=800)
        clf.fit(X_train, y_train, sample_weight= np.ascontiguousarray(w_train) )
        dump(clf, svm_file)
    else:
        print('will read the classifier from file')
        clf = load(svm_file)

    print('classified ready')

    prob_train = clf.predict_proba(X_train)
    prob_test = clf.predict_proba(X_test)

    #print('test prob calculated', prob_test)

    ## plot the probability distributions for signal and background

    prob_test_signal = prob_test[:,1][ y_test==1 ]
    prob_test_background = prob_test[:,1][ y_test==0 ]

    prob_train_signal = prob_train[:,1][ y_train==1 ]
    prob_train_background = prob_train[:,1][ y_train==0 ]


    bins = np.linspace(0.,0.4, 40)

    #plt.clf()
    #plt.hist(prob_test_background, bins=bins, density=True, label=['background'])
    #plt.hist(prob_test_signal, bins=bins, density=True, label=['signal'], histtype=u'step')

    #plt.xlabel('Probability of signal')
    #plt.title(label+' unweighted frames, test sample')
    #plt.ylabel("# Entries (Norm)")
    #plt.legend() # (loc='upper right')
    #plt.savefig("prob_distribution_"+tag+"_unweighted.png")
    #plt.show()

    ## same but weighted frames now

    w_test_signal = w_test[ y_test==1  ]
    w_test_background = w_test[ y_test==0  ]

    w_train_signal = w_train[ y_train==1  ]
    w_train_background = w_train[ y_train==0  ]

    makeSignalBackPlot(prob_test_background, w_test_background, prob_test_signal, w_test_signal, bins, 'Probability of signal', 'Entries (Norm.)', lab_signal, label+' weighted frames, test sample', "prob_distribution_"+tag)
    

    bins1 = np.linspace(0.,1., 100)

    bkg_values, bin_edges = np.histogram(prob_test_background, bins=bins1, weights=w_test_background, density=True)
    sig_values, _ = np.histogram(prob_test_signal, bins=bins1, weights=w_test_signal, density=True)

    bkg_values_train, bin_edges_train = np.histogram(prob_train_background, bins=bins1, weights=w_train_background, density=True)
    sig_values_train, _ = np.histogram(prob_train_signal, bins=bins1, weights=w_train_signal, density=True)

    #############################################
    ###### combine both test and train datasets
    #############################################

    prob_bkg_all = np.concatenate((prob_train_background,prob_test_background))
    weight_bkg_all = np.concatenate( (w_train_background, w_test_background))
    prob_sig_all = np.concatenate((prob_train_signal,prob_test_signal))
    weight_sig_all = np.concatenate( (w_train_signal, w_test_signal))

    limit = calcLimitFromHisto(prob_bkg_all, weight_bkg_all, prob_sig_all, weight_sig_all, bins1)
    
    bkg_values_all, bin_edges_all = np.histogram(prob_bkg_all, bins=bins1, weights=weight_bkg_all, density=True)
    sig_values_all, bin_edges_all = np.histogram(prob_sig_all, bins=bins1, weights=weight_sig_all, density=True)

    ## calculate the limit here
    #N_norm_bkg = bkg_values_all.sum()
    #N_norm_sig = sig_values_all.sum()
    #bkg_values_all_norm = bkg_values_all * N_bkg_all/ N_norm_bkg
    #sig_values_all_norm = sig_values_all * N_sig_all/ N_norm_sig

    #limit_old = getLimit(bkg_values_all_norm, sig_values_all_norm)
    #print(r'the (old) limit is: $\times$ {:.2f} '.format( limit_old) )
    print(r'the limit is: $\times$ {:.2f} '.format( limit) )

    plt.clf()
    #bins = np.linspace(0.,1., 50)
    plt.hist(prob_bkg_all, bins=bins, weights=weight_bkg_all, label=['background'], density=True)
    plt.hist(prob_sig_all, bins=bins, weights=weight_sig_all, label=['signal '+lab_signal], histtype=u'step', density=True)

    plt.xlabel('Probability of signal')
    plt.title(label+' weighted frames, test+train sample')
    plt.ylabel("# Entries")
    plt.legend() # (loc='upper right')
    plt.savefig("prob_distribution_"+tag+"_all.png")

    plt.yscale('log')
    plt.text(0.05, 50, r'limit  $\times${:.2f} '.format( limit))
    plt.savefig("prob_distribution_"+tag+"_all_log.png")


    #print('test hist bkg integral:', bkg_values.sum())
    #print('test hist bkg integral:', sig_values.sum())

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

    #plt.ylim(0.01,1)
    #plt.xlim(0.01,1)
    #plt.yscale('log')
    #plt.xscale('log')
    #plt.savefig("ROC_weighted"+tag+"_log.png")
    #plt.show()

    ## ROC curves
    #print (y_test[0:100])
    #print(prob_test[:, 1])

    fpr, tpr, thresholds = roc_curve(y_test, prob_test[:, 1])
    roc_auc = auc(fpr, tpr)

    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, prob_train[:, 1])
    roc_auc_train = auc(fpr_train, tpr_train)

    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f) test' % roc_auc)
    plt.plot(fpr_train, tpr_train, label='ROC curve (area = %0.2f) train' % roc_auc_train)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC unweighted: '+label)
    plt.legend(loc="lower right")
    plt.savefig("ROC_"+tag+".png")
    #plt.show()

    print('-----> Studied signal '+tag+' limit {:.2f} compare against standard {:.2f} '.format( limit, standard_limit))
    return limit, standard_limit
    
#dict_signals={ 1:'2', 2:'10', 3:'20', 4:'30', 5:'40', 6:'50', 7:'60', 8:'70'}
##
## Usage:
##  python svm_clas.py  <doTraining: 0/1>  sig1 sig2 sig3 ...
##

''' In original code doTrain = False and tag_signal = [], there is no training and no specified tag signal,
    This seems to be the key to making the code work'''
    
if __name__ == "__main__":
    doTrain = True
    tag_signal = [4]

    if len(sys.argv) > 1:
        doTrain = int(sys.argv[1])
    if len(sys.argv) > 2:
        for j in range(2, len(sys.argv)):
            tag_signal.append( int(sys.argv[j]) )

    print(doTrain, tag_signal)

    v_limit = []
    v_limit_standard = []
    v_mass = []
    for j in tag_signal:
        limit, standard_limit = main(doTrain, j)
        v_limit.append(limit)
        v_limit_standard.append(standard_limit)
        v_mass.append(int(dict_signals[j]))

    print('Limits are:')
    print(v_mass)
    print(v_limit)
    print(v_limit_standard)
