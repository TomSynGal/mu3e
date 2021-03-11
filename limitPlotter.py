import numpy as np
import matplotlib.pyplot as plt


def plotLimit(xval, yval, yval1):
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(xval, yval, '--bo',label='Support vector machine')
    plt.plot(xval, yval1, '-k.',label=r'$m_{ee}^{both}$ fit')
    
    plt.xlabel('Dark photon mass [MeV]')
    plt.ylabel(r'Br$(\mu\to e (A\to ee) \nu\nu) \times 10^{-10}$')
    plt.title('Dark photon upper limit comparisons')
    ax.text(0.6,0.75,'Mu3e work in progress',transform=ax.transAxes)
    ax.text(0.6,0.7,'95% CL upper limits',transform=ax.transAxes)
    plt.legend()
    plt.gca().xaxis.grid(True)
    plt.gca().yaxis.grid(True)

    plt.savefig("limit.png")
    plt.yscale('log')
    plt.ylim((0.1,10.))
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')

    plt.savefig("limit_log.png")





#      1      2      3       4       5      6   7       8   
xval =[2,     10,    20,     30,     40,   50,  60,     70]

yval = [4.674544781781934, 0.7691049742379429, 4.030831764727988,2.9757232793983355, 1.5483816881397974, 0.936177514933769,0.8506124703526026, 0.3118866610960934]

yval1 = [25.33421759964665, 12.344106956922083, 11.43028825529753,2.4784164372315205, 1.1139860224616485, 0.5380935017208495,0.31945661975843154, 0.15490476983793153]



plotLimit(xval, yval, yval1)

"""
 notes







"""
