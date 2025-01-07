# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:05:44 2022

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
rounds=10
w=1000
s_1=0.0
s_2=0.0
s_3=0.0

psi_0=basis(2,0)
psi_1=basis(2,1)
psi_xp=(psi_0+psi_1).unit()
psi_yp=(psi_0+1j*psi_1).unit()
rho_xp=psi_xp*psi_xp.dag()
rho_yp=psi_yp*psi_yp.dag()
def E_noise(t):
    s=s_1*np.cos(omega_n1*(t))+s_2*np.cos(omega_n2*(t))+s_3*np.cos(omega_n3*(t))
    return(s)
# def Omega_I(t,arg):
#     phi_0=arg['phi_0']
#     s_t=arg['A_x']
#     omega_t=s_t*np.sin(omega_q*(1)*(t+phi_0))
#     return omega_t

# A_x=5e-3*2*np.pi
# t_0=0e3
# arg_1={'phi_0':0,'A_x':A_x,'t_0':t_0}
# H=[[sigmaz()*omega_q/2,E_noise]]
# result=mesolve(H,psi_p,t_range,[],[],args=arg_1)
def multijob_RTO_noise():
    P_x=[]
    P_y=[]
    Shot_x=[]
    Shot_y=[]
    for i in range(N):
        t_x=2*i*d_t
        t_y=(2*i+1)*d_t
        phi_x=E_noise(t_x)
        # psi_x=(basis(2,0)+np.exp(1j*phi_x)*basis(2,1)).unit()
        # rho_x=psi_x*psi_x.dag()
        # p_x=(rho_xp*rho_x).tr()
        # p_x=np.cos(phi_x)/2+1/2
        p_x=1/2-np.sin(phi_x)/2
        P_x.append(p_x) 
        shot_x=random.choices([0,1],weights=[p_x,1-p_x])[0]
        Shot_x.append(shot_x)
        phi_y=E_noise(t_y)
        # psi_y=(basis(2,0)+np.exp(1j*phi_y)*basis(2,1)).unit()
        # rho_y=psi_y*psi_y.dag()
        # p_y=(rho_yp*rho_y).tr()
        p_y=1/2-np.sin(phi_y)/2
        # p_y=np.cos(phi_y)/2+1/2
        # p_y=(np.sqrt(1-(2*p_yx-1)**2)+1)/2
        P_y.append(p_y) 
        shot_y=random.choices([0,1],weights=[p_y,1-p_y])[0]
        Shot_y.append(shot_y)
    
    # Phi_yy=[]
    Phi_r=[]
    T=[]
    for i in range(int(N-w+1)):
        t_i=2*(i)*d_t
        p_xy=1-np.mean(Shot_x[i:i+w])
        # p_x=(np.sqrt(1-(2*p_xy-1)**2)+1)/2
        p_xy=P_x[i]
        # p_x=1-(Shot_x[i])
        # phi_x=np.arccos(2*p_x-1)
        # p_y=1-np.mean(Shot_y[i:i+w])
        # p_y=(np.sqrt(1-(2*p_yx-1)**2)+1)/2
        # p_y=1-(Shot_y[i])
        p_y=P_y[i]
        # phi_y=np.arcsin((1-2*p_y))
        # Phi_yy.append(phi_y)
        phi_r=np.angle((2*p_x-1)+1j*(1-2*p_y))/500e-9
        # Phi_x.append(phi_x)
        # phi_r=np.arctan((1-2*p_y)/(2*p_x-1))
        Phi_r.append(phi_r)
        T.append(t_i)
    return(Phi_r)
data=Parallel(n_jobs=12, verbose=2)(delayed(multijob_RTO_noise)() for k in range(rounds))
Phi_mat=np.array(data)

Phi_r=[]
for i in range(rounds):
    Phi_r=Phi_r+data[i]
Phi_k=[]
F=[(i-int((N-w)*rounds)/2)/(2*d_t*(N-w+1)*rounds) for i in range(int((N-w+1)*rounds))]
# Phi=[np.arctan(np.sin(E_noise(2*i*d_t))/np.cos(E_noise((2*i+1)*d_t))) for i in range(int(N/2-w+1))]
# Phi=[np.arctan((1-2*P_y[i])/(2*P_x[i]-1)) for i in range(int(N/2))]
# for i in range(int(N/2-2)):
#     k=i-(N/2-3)/2
#     f_k=k/(2*d_t*(N/2-2))
#     phi_kx=0
#     phi_ky=0
#     for m in range(int(N/2-2)):
#         j=m
#         # phi_kx+=Phi_x[j]*np.exp(-1j*2*np.pi/(N/2)*j*k)
#         phi_ky+=Phi_y[m]*np.exp(-1j*2*np.pi/(N/2-2)*j*k)
#     Phi_k.append(phi_ky*np.conjugate(phi_ky)/(N/2-2)**2*4)
#     F.append(f_k)
Phi_f=np.fft.fftshift((np.fft.fft(Phi_r)))
Phi_fs=(2*np.pi)**2*Phi_f*np.conjugate(Phi_f)*(400e-6)/(N-w)/rounds
plt.figure()
# plt.plot(T,[Phi_y[i]-Phi_x[i] for i in range(len(T))])
plt.plot(F[int(N*rounds/2):int(N*rounds-1)],Phi_fs[int(N*rounds/2):int(N*rounds-1)])
plt.xscale('log')
plt.yscale('log')
# plt.plot(T,[(np.sin(np.cos(omega_n*i))) for i in T])
# psi_t=result.states[-1]
# P_0=result.expect[0]

# plt.figure()
# plt.plot(T,Phi_yy,label=r'$P_0$')
# plt.plot([2*i*d_t for i in range(int(N/2))],P_r,label=r'$P_0$')
# plt.plot(t_range,P_1,label=r'$P_1$')
endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')