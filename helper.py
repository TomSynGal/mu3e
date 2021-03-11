import numpy as np
import scipy.stats
from math import sqrt, isinf
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

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

def getLimit(hbkg, hsig, confidenceLevel=0.95, doIt=True):
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


def makeSignalBackPlot(x_bkg, w_bkg, x_sig, w_sig, bins, xlabel, ylabel, lab_signal, title, pname, doLog=False):
    plt.clf()
    dens = True
    h_main,_,_=plt.hist(x_bkg, bins=bins,weights=w_bkg,label=['background'],density=dens, color='aqua')
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
    plt.errorbar(bincenters, h_main,barsabove=True, ls='', yerr=h_main_err, marker='+',color='deepskyblue')
    
    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend() 
    plt.savefig(pname+".png")
    if doLog:
        plt.yscale('log')
        plt.savefig(pname+"_log.png")


    

def calcLimitFromHisto(x_bkg, w_bkg, x_sig, w_sig, bins):
    N_bkg_all = w_bkg.sum()
    N_sig_all = w_sig.sum()
    bkg_val, bin_edges = np.histogram(x_bkg, bins=bins, weights=w_bkg, density=True)
    sig_val, bin_edges = np.histogram(x_sig, bins=bins, weights=w_sig, density=True)

    ## calculate the limit here
    N_norm_bkg = bkg_val.sum()
    N_norm_sig = sig_val.sum()
    bkg_values_all_norm = bkg_val * N_bkg_all/ N_norm_bkg
    sig_values_all_norm = sig_val * N_sig_all/ N_norm_sig

    limit = getLimit(bkg_values_all_norm, sig_values_all_norm)
    return limit
