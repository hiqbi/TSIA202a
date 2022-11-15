# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:31:54 2022

@author: IQBI Hamza
"""

import numpy as np
from scipy import signal as sig
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.widgets as wid
import randproc as rp



def f() :
    H = 30      # number of points for covariance
    n = 2*H -1  # number of process' samples
    tc = np.arange(-(H-1),H)  # temporal axis for ACF
    m = 10*n    # The size of the computed DTFT
    
    # Generate an AR(1) process
    Z = np.random.normal(0,1,n) 
    phi1  = 0.6
    Pcoeffs = np.array([1.])                 
    Qcoeffs = np.poly((phi1,))
    AR = sig.lfilter(Pcoeffs, Qcoeffs,Z)
    mu_AR = np.mean(AR)
    I_AR = np.abs(np.fft.fft(AR - mu_AR, m))**2 / (2*np.pi*n)
    
    # Generate a Harmonic process
    A_0 = 1
    omega = np.pi/3
    phi =   np.pi +2*np.pi*np.random.random_sample()
    SIN = A_0 * np.cos(omega*tc+phi) +  np.random.normal(0,1,n)
    mu_SIN = np.mean(SIN)
    I_SIN = np.abs(np.fft.fft(SIN - mu_SIN, m))**2 / (2*np.pi*n)
    
    # Generate a White Noise WN(0, 1)
    sigma = 1
    WN = np.random.normal(0,sigma,n)
    mu_WN = np.mean(WN)
    I_WN = np.abs(np.fft.fft(WN - mu_WN, m))**2 / (2*np.pi*n)
    
    # Plot the periodogram of the 3 processes at once
    lambdas = np.arange(0, m)*2*np.pi/m
    plt.plot(lambdas, I_WN)
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$I_n(\lambda)$')
    plt.show()
    


N = 50
variances = []
for n in np.logspace(1, 4, 20, dtype=int) :
    var_n = []
    for k in range(N) :
        X = np.random.normal(0, 1, n)
        gamma = rp.acovb(X)
        v = np.var(gamma)
        var_n.append(v)
    variances.append((sum(var_n))/len(var_n))

plt.plot(np.logspace(1, 4, 20, dtype=int), variances)
plt.xlabel('n', fontsize=30)
plt.ylabel(r'$Var(\widehat{\gamma}_n)$', fontsize=30)