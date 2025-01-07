# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 22:16:03 2022

@author: mantoutou
"""



from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
starttime=int(time.time())


#%%
N=30
E_C=1.41455256018918  #GHZ
E_J=79.9527706058577 #GHZ
num=201
sigma_0=3
g=1/np.sqrt(2*np.pi)/sigma_0*1.019 #0.0125*2*np.pi,1.019来自Q_zpf的偏差

T=24 #18结果更好
t_range=np.linspace(0,T,num) #ns
N_g=np.linspace(-2,2,num)
n_range=np.linspace(0,N,N+1)
n_g=0
a_d=create(N)
a=destroy(N)
# Omega=(E_J/E_C/8)**0.25/np.sqrt(2)
# lambda_0=0.5/2.976
# t_0=T/2

n_zpf=(E_J/(32*E_C))**0.25
phi_zpf=(2*E_C/E_J)**0.25
n_oper=-n_zpf*1j*(a-a_d)
phi_oper=phi_zpf*(a+a_d)
sin_halfPhi=(phi_oper/2)-(phi_oper/2)**3/6+(phi_oper/2)**5/120
cos_halfPhi=1-(phi_oper/2)**2/2+(phi_oper/2)**4/24-(phi_oper/2)**6/720
# H_t=4*E_C*n_oper**2+E_J*phi_oper**2/2-E_J/24*phi_oper**4+E_J/720*phi_oper**6-E_J/40320*phi_oper**8

H_t=np.sqrt(8*E_J*E_C)*(a_d*a)-(a+a_d)**4*E_C/12+(a+a_d)**6*E_C/360*(2*E_C/E_J)**0.5

# Q_zpf=np.sqrt(1/(2*np.sqrt(8*E_J*E_C)))
Q_zpf=(E_J/E_C/8)**0.25/np.sqrt(2)
H_n=-1j*Q_zpf*(a-a_d)

E_0=H_t.eigenenergies()[0]
E_1=H_t.eigenenergies()[1]
E_2=H_t.eigenenergies()[2]
psi_0=H_t.eigenstates()[1][0]
psi_1=H_t.eigenstates()[1][1]
psi_2=H_t.eigenstates()[1][2]
rho_0=psi_0*psi_0.dag()
rho_1=psi_1*psi_1.dag()
rho_2=psi_2*psi_2.dag()
omega_0=E_1-E_0
omega_1=E_2-E_1
omega_02=(omega_0+omega_1)/2
alpha=omega_0-omega_1
print(omega_0/2/np.pi)

s_01=(psi_1*psi_0.dag()*sin_halfPhi).tr()
s_12=(psi_1*psi_2.dag()*sin_halfPhi).tr()
c_00=(psi_0*psi_0.dag()*cos_halfPhi).tr()
c_11=(psi_1*psi_1.dag()*cos_halfPhi).tr()
c_22=(psi_2*psi_2.dag()*cos_halfPhi).tr()
x_qp=1e-6
e=1.6021766208e-19
h=6.626070154e-34
k_b=1.3806e-23
T_e=20e-3
hbar=h/(2*np.pi)
I_c=(E_J*1e9)*2*e
L_J=h/(2*np.pi*2*e*I_c)
Delta_Al=230e-6*e
g_T=hbar/(L_J*np.pi*Delta_Al) #电导率
Re_Yqp=0.5*x_qp*g_T*(2*Delta_Al/abs(hbar*omega_0*1e9))**1.5
g_K=e**2/h
# Gamma=(psi_1.dag()*H_sinPhi2(n_g)*psi_0)*omega_0*1e9/np.pi/g_K*Re_Yqp/2/np.pi
S_qp=omega_0*1e9/np.pi/g_K*Re_Yqp/np.tanh(h*omega_0*1e9/2/k_b/T_e)
Gamma=s_01**2*S_qp
# Gamma=np.sqrt(2*omega_0*1e9*Delta_Al/hbar)/np.pi*x_qp



#%%
T_e=20e-3
Gamma_p=18
Gammap_11=2*c_11**2*Gamma_p/(c_11**2+c_22**2)
Gamma_10=Gammap_11*s_01**2/c_11**2*np.sqrt(np.pi*Delta_Al**2/(k_b*T_e*hbar*omega_0*1e9))
T_1=1/Gamma_10
Re_Yqp_0=0.5*g_T*(2*Delta_Al/abs(hbar*omega_0*1e9))**1.5
S_qp_0=omega_0*1e9/np.pi/g_K*Re_Yqp_0/np.tanh(h*omega_0*1e9/2/k_b/T_e)
x_qp=Gamma_10/s_01**2/S_qp_0
print(T_1,x_qp)
# Gammap_22=2*c_22**2*Gamma_p/(c_11**2+c_22**2)






