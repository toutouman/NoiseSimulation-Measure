

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 09:51:03 2022

@author: mantoutou
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
import random
from scipy.fftpack import fft
starttime=int(time.time())

num=1001
T=100e-9 #18结果更好
t_range=np.linspace(0,T,num) #
omega_q=5e9*2*np.pi
omega_n1=100*2*np.pi
omega_n2=200*2*np.pi
omega_n3=1000*2*np.pi
d_t=200e-6
N=100000
rounds=100
w=10
s_1=0.1
s_2=0.0
s_3=0.0

psi_0=basis(2,0)
psi_1=basis(2,1)
psi_xp=(psi_0+psi_1).unit()
psi_yp=(psi_0+1j*psi_1).unit()
rho_xp=psi_xp*psi_xp.dag()
rho_yp=psi_yp*psi_yp.dag()
def E_noise(t):
    s=(random.random()-0.5)*0
    return(s)

def multijob_RTO_noise(N,w):
    P_x=[]
    P_y=[]
    Shot_x=[]
    Shot_y=[]
    Phi_x=[]
    for i in range(N):
        t_x=i*d_t
        t_y=i*d_t
        phi_x=E_noise(t_x)
        Phi_x.append(phi_x)
        # psi_x=(basis(2,0)+np.exp(1j*phi_x)*basis(2,1)).unit()
        # rho_x=psi_x*psi_x.dag()
        # p_x=(rho_xp*rho_x).tr()
        # p_x=np.cos(phi_x)/2+1/2
        p_x=np.cos(phi_x)/2+1/2
        P_x.append(p_x) 
        shot_x=random.choices([0,1],weights=[p_x,1-p_x])[0]
        Shot_x.append(shot_x)
        # phi_y=E_noise(t_y)
        # psi_y=(basis(2,0)+np.exp(1j*phi_y)*basis(2,1)).unit()
        # rho_y=psi_y*psi_y.dag()
        # p_y=(rho_yp*rho_y).tr()
        p_y=1/2-np.sin(phi_x)/2
        # p_y=np.cos(phi_y)/2+1/2
        # p_y=(np.sqrt(1-(2*p_yx-1)**2)+1)/2
        P_y.append(p_y) 
        shot_y=random.choices([0,1],weights=[p_y,1-p_y])[0]
        Shot_y.append(shot_y)
    
    # Phi_yy=[]
    Phi_r=[]
    T=[]
    for i in range(int(N/w)):
        t_i=(i)*d_t
        p_xy=1-np.mean(Shot_y[i*w:i*w+w])
        p_x=(np.sqrt(1-(2*p_xy-1)**2)+1)/2
        # p_x=np.mean(P_x[i*w:i*w+w])
        # p_x=1-(Shot_x[i])
        # phi_x=np.arccos(2*p_x-1)
        # p_y=1-np.mean(Shot_y[i:i+w])
        # p_y=np.mean(P_y[i*w:i*w+w])
        # p_y=(np.sqrt(1-(2*p_yx-1)**2)+1)/2
        p_y=1-np.mean(Shot_y[i*w:i*w+w])
        # phi_y=np.arcsin((1-2*p_y))
        # Phi_yy.append(phi_y)
        phi_r=np.angle((2*p_x-1)+1j*(2*p_y-1))/500e-9
        Phi_r.append(phi_r)
        T.append(t_i)
    return(Phi_r)

window_list=[1,10,100,1000]
Phi_FS=[]
plt.figure()
for j in range(len(window_list)):
    w=window_list[j]
    data=Parallel(n_jobs=12, verbose=2)(delayed(multijob_RTO_noise)(N,w) for k in range(rounds))
    Phi_mat=np.array(data)
    Phi_r=[]
    for i in range(rounds):
        Phi_r=Phi_r+data[i]
    Phi_k=[]
    F= np.linspace(-1/d_t/w/2,1/d_t/w/2,int((N/w)*rounds))
    Phi_f=np.fft.fftshift((np.fft.fft(Phi_r)))
    # Phi_fs=(2*np.pi)**2*Phi_f*np.conjugate(Phi_f)*(400e-6)/(N-w)/rounds
    Phi_fs=(2*np.pi)**2*Phi_f*np.conjugate(Phi_f)*w*d_t/((N/w)*rounds)
    Phi_FS.append(Phi_fs)
    # plt.plot(T,[Phi_y[i]-Phi_x[i] for i in range(len(T))])
    plt.plot(F[int(N/w*rounds/2+1):int(N/w*rounds-1)],Phi_fs[int(N/w*rounds/2+1):int(N/w*rounds-1)])
plt.xscale('log')
plt.yscale('log')

endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')