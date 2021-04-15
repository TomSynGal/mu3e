# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:04:05 2021

@author: thoma
"""


import numpy as np
import scipy.stats
import itertools
from math import sqrt, isinf
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def SignalBackgroundPlot(x_bkg, w_bkg, x_sig, w_sig, bins, xlabel, ylabel, lab_signal, title, pname, doLog=False):
    plt.clf()
    dens = True
    h_main,_,_=plt.hist(x_bkg, bins=bins,weights=w_bkg,label=['Background'],density=dens, color='aqua')
    plt.hist(x_sig, bins=bins,weights=w_sig,label=['Signal '+lab_signal], histtype=u'step', density=True, color='k')

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
    plt.errorbar(bincenters, h_main,barsabove=True, ls='', yerr=h_main_err, marker='+',color='deepskyblue')
    
    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend() 
    plt.savefig(pname+".png")
    plt.show()
    if doLog:
        plt.yscale('log')
        plt.savefig(pname+"_log.png")
  
##### make plot
def MakePlot(X1,X2, tag, Nb, **kwargs):
    plt.clf()
    plt.grid()
    
    xtitle=tag
    title = tag
    for key, value in kwargs.items():
        if key == "xtitle":
            xtitle = value
        elif key=="title":
            title = value
        

    themin = min( [min(X1), min(X2)])
    themax = max( [max(X1), max(X2)])
    bins = np.linspace(themin, themax, Nb)

    plt.hist(X1, bins=bins, density=True, label=['background'])
    plt.hist(X2, bins=bins, density=True, label=['signal'], histtype=u'step')

    plt.xlabel(xtitle)
    plt.title(title)
    plt.ylabel("Normalised Entries")
    plt.legend(loc='upper right')
    plt.savefig(tag+".png")
    
##### make plot
def MakeLogPlot(X1,X2, tag, Nb, **kwargs):
    plt.clf()
    plt.grid()
    plt.yscale('log')
    
    xtitle=tag
    title = tag
    for key, value in kwargs.items():
        if key == "xtitle":
            xtitle = value
        elif key=="title":
            title = value
        

    themin = min( [min(X1), min(X2)])
    themax = max( [max(X1), max(X2)])
    bins = np.linspace(themin, themax, Nb)

    plt.hist(X1, bins=bins, density=True, label=['background'])
    plt.hist(X2, bins=bins, density=True, label=['signal'], histtype=u'step')

    plt.xlabel(xtitle)
    plt.title(title)
    plt.yscale('log')
    plt.ylabel("Normalised Entries")
    plt.legend(loc='upper right')
    plt.savefig(tag+".png")    
        
        
def GetLimit(hbkg, hsig, confidenceLevel=0.95, doIt=True):
    N = len(hbkg)
    ns, nb = 0., 0.
    res = 0.
    for j in range(N):
        i = N - j - 1
        ns += hsig[i]
        nb += hbkg[i]
        if nb > 3:
            sign = ns/sqrt(nb)
            res += sign*sign
            #print('bin ',i, ns, nb, sign, res)
            ns, nb = 0., 0.
        else:
            continue
            
    s = scipy.stats.norm.ppf(1-(1-confidenceLevel)*0.5)
    lim = s/sqrt(res)
    #if isinf(lim) and doIt:
    #    return getLimit(savgol_filter(hbkg,5,2), savgol_filter(hsig,5,2),
    #                    confidenceLevel, False)
        
    
    return lim

def GetLimit90(hbkg, hsig, confidenceLevel=0.90, doIt=True):
    N = len(hbkg)
    ns, nb = 0., 0.
    res = 0.
    for j in range(N):
        i = N - j - 1
        ns += hsig[i]
        nb += hbkg[i]
        if nb > 3:
            sign = ns/sqrt(nb)
            res += sign*sign
            #print('bin ',i, ns, nb, sign, res)
            ns, nb = 0., 0.
        else:
            continue
            
    s = scipy.stats.norm.ppf(1-(1-confidenceLevel)*0.5)
    lim = s/sqrt(res)
    #if isinf(lim) and doIt:
    #    return getLimit(savgol_filter(hbkg,5,2), savgol_filter(hsig,5,2),
    #                    confidenceLevel, False)
        
    
    return lim

        
def CalculateHistogramLimits(x_bkg, w_bkg, x_sig, w_sig, bins):
    N_bkg_all = w_bkg.sum()
    N_sig_all = w_sig.sum()
    bkg_val, bin_edges = np.histogram(x_bkg, bins=bins, weights=w_bkg, density=True)
    sig_val, bin_edges = np.histogram(x_sig, bins=bins, weights=w_sig, density=True)

    ## calculate the limit here
    N_norm_bkg = bkg_val.sum()
    N_norm_sig = sig_val.sum()
    bkg_values_all_norm = bkg_val * N_bkg_all/ N_norm_bkg
    sig_values_all_norm = sig_val * N_sig_all/ N_norm_sig

    limit = GetLimit(bkg_values_all_norm, sig_values_all_norm)
    return limit

def CalculateHistogramLimits90(x_bkg, w_bkg, x_sig, w_sig, bins):
    N_bkg_all = w_bkg.sum()
    N_sig_all = w_sig.sum()
    bkg_val, bin_edges = np.histogram(x_bkg, bins=bins, weights=w_bkg, density=True)
    sig_val, bin_edges = np.histogram(x_sig, bins=bins, weights=w_sig, density=True)

    ## calculate the limit here
    N_norm_bkg = bkg_val.sum()
    N_norm_sig = sig_val.sum()
    bkg_values_all_norm = bkg_val * N_bkg_all/ N_norm_bkg
    sig_values_all_norm = sig_val * N_sig_all/ N_norm_sig

    limit = GetLimit90(bkg_values_all_norm, sig_values_all_norm)
    return limit

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

def plot_confusion_matrix (cm, classes, confusionName,
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
    plt.savefig(confusionName+".png")

def PlotLimit(xval, yval, label0, yval1, label1, savetag, confidencetag):
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(xval, yval, '--bo',label=label0)
    plt.plot(xval, yval1, '-k.',label=label1)
    
    plt.xlabel('Dark photon mass [MeV]')
    plt.ylabel(r'Br$(\mu\to e (A\to ee) \nu\nu) \times 10^{-10}$')
    plt.title('Dark photon upper limit comparisons')
    ax.text(0.6,0.75,'Mu3e',transform=ax.transAxes)
    ax.text(0.6,0.7,confidencetag+'% CL upper limits',transform=ax.transAxes)
    plt.legend()
    plt.gca().xaxis.grid(True)
    plt.gca().yaxis.grid(True)

    plt.savefig("limit.png")
    plt.yscale('log')
    plt.ylim((0.01,50.))
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')

    plt.savefig(savetag+"_limit_log.png")

def Plot3Limit(xval, yval, label0, yval1, label1, yval2, label2, savetag, confidencetag):
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(xval, yval, '--bo', label= label0)
    plt.plot(xval, yval1, '-k.', label= label1)
    plt.plot(xval, yval2, '--r', label = label2)
    
    plt.xlabel('Dark photon mass [MeV]')
    plt.ylabel(r'Br$(\mu\to e (A\to ee) \nu\nu) \times 10^{-10}$')
    plt.title('Dark photon upper limit comparisons')
    ax.text(0.6,0.45,'Mu3e',transform=ax.transAxes)
    ax.text(0.6,0.40,confidencetag+'% CL upper limits',transform=ax.transAxes)
    plt.legend()
    plt.gca().xaxis.grid(True)
    plt.gca().yaxis.grid(True)

    plt.savefig("limit.png")
    plt.yscale('log')
    plt.ylim((0.01,50.))
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')

    plt.savefig(savetag+"_limit_log.png")

#Include weights to confusion matrix or explain why you haven't used them.
#Not too important to include weights but do roc curves and limits first.