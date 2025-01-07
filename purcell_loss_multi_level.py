# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 22:18:02 2023

@author: 馒头你个史剃磅
"""


from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time

omega_RR=0.3
A=0.8
kappa_n=0.1
Omega_r=np.linspace(3,6,101)
N=30


def loss_n(omega_q,n):
    return(A*(n*omega_RR)*np.sqrt(n)*
           kappa_n/(2*(omega_q-n*omega_RR)**2+n*kappa_n**2))/omega_q
Loss_r=[]
T1=[]
for omega_q in Omega_r:
    loss_f=0
    for n in range(N):
        loss_f=loss_f+loss_n(omega_q,n+1)
    Loss_r.append(loss_f)
    T1.append(1/(loss_f))
plt.figure()
plt.plot(Omega_r,T1)
#%%
omega_RR=6.5e9*2*np.pi
A=60e6*2*np.pi
kappa_n=1*1e6*2*np.pi
Omega_r=np.linspace(2.5,6,101)*1e9*2*np.pi
N=30


def loss_purcell(omega_q):
    return(A**2*(omega_RR)*kappa_n/(2*(omega_q-omega_RR)**2+kappa_n**2))/omega_q
def loss_purcell2(omega_q):
    return(A*(omega_RR)*kappa_n/(2*(omega_q-omega_RR)**2+kappa_n**2))/omega_q
Loss_r=[]
T1=[]
for omega_q in Omega_r:
    loss_f=0
    loss_f=loss_f+loss_purcell(omega_q)
    Loss_r.append(loss_f)
    T1.append(1/(loss_f))
plt.figure()
plt.plot(Omega_r/2/np.pi,T1)
