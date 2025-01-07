# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:29:22 2023

@author: mantoutou
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import optimize
import sys
import time
sys.path.append(r'C:\Users\mantoutou\OneDrive\文档\程序\科大\噪声仿真&测试\CPMG分析')
from CPMG_filter_function import Noise_Teff, CPMG_Tphi

#%% 计算不同衰减器下的等效温度
NT_1=Noise_Teff(
            AT_RT = 0+6,      #室温层衰减器 dB
            AT_50K = 1+4,     #50K层衰减器 dB
            AT_4K=10+0.5,      #4K层衰减器 dB
            AT_Still=10+0.5,   #Still层衰减器 dB
            AT_CP=20+0.5,     #100mK层衰减器 dB
            AT_MC=30+0.5,      #MC层衰减器 dB
            T_Singal = 1819,  #信号噪声温度 K
            )
A=NT_1.Svv_all_func(6.2e9*2*np.pi)

# omega_list = np.linspace(-10e6*2*np.pi,10e6*2*np.pi,101)
# S_list = [NT_1.Svv_f_func(o) for o in omega_list]
print(A)
# plt.figure()
# plt.plot(omega_list/2/np.pi,S_list)
#%%
NT_1=Noise_Teff(
            AT_RT = 20,      #室温层衰减器 dB
            AT_50K = 1.05,     #50K层衰减器 dB
            AT_4K=14.78,      #4K层衰减器 dB
            AT_Still=0.78,   #Still层衰减器 dB
            AT_CP=0.64,     #100mK层衰减器 dB
            AT_MC=2.34,      #MC层衰减器 dB
            T_Singal = 300,  #信号噪声温度 K
            )
A=NT_1.Svv_all_func(5e9*2*np.pi)
print(A)
#%%
omega_r = 0.44e9*2*np.pi
T_eff = 0.009
h=6.626e-34
h_bar=h/2/np.pi
k_b=1.38065e-23
n_bar=1/(np.exp(h_bar*omega_r/k_b/T_eff)-1)
print(n_bar)
#%% 在给定等效温度下计算由于光子数限制的比特T2

"""定义热光场态产生的噪声谱线"""
def S_th(omega_noise,kappa,T_eff,g,
         omega_r = 6.5e9*2*np.pi,
         omega_q = 5.5e9*2*np.pi,
         E_C = 220e6*2*np.pi):
    f_noise=omega_noise/2*np.pi
    """热态平均光子数"""
    h=6.626e-34
    h_bar=h/2/np.pi
    k_b=1.38065e-23
    n_bar=1/(np.exp(h_bar*omega_r/k_b/T_eff)-1)
    # print(n_bar)
    
    """光子数涨落的噪声谱 单位 Hz^-1 """
    s_th=(n_bar**2+n_bar)*2*kappa/(omega_noise**2+kappa**2)

    Delta=omega_q-omega_r
    """比特色散位移  单位 HZ"""
    chi_g=g**2/(Delta*(1-Delta/E_C))
    # print(chi_g)
    # print(E_C/2/np.pi,Delta/2/np.pi,chi_g/2/np.pi)
    # chi_g=0.3e6*2*np.pi
    eta=kappa**2/(kappa**2+4*chi_g**2)
    """比特频率涨落噪声谱  单位 HZ"""
    s_f=(2*chi_g)**2*s_th*eta/2/np.pi
    # print(chi_g)
    return(s_f)




kappa=8e6*2*np.pi
T_eff=0.100
g=120e6*2*np.pi
omega_r=6.5e9*2*np.pi
omega_q=5.5e9*2*np.pi
# tau=20

N_list=[1]

t_list=np.linspace(0.1e-6,20e-6,40)
cal_Tphis=[]
Cal_T2_by_Sw_list=[]

plt.figure()
for i in range(len(N_list)):
    N=N_list[i]   
    starttime_i=int(time.time())
    
    cal_Tphi_n = CPMG_Tphi(
                  S_w = S_th,
                  N = N,
                  tau_list = t_list,
                  omega_s = 0.1*2*np.pi,
                  omega_e = 20e6*2*np.pi,
                  int_limit = 20e3, 
                  t_gate = 0e-9,
                  args = {'Sw_args': [kappa,T_eff,g,omega_r,omega_q]}, 
                  )
    cal_Tphis.append(cal_Tphi_n)

    P_N,Interg,T2_e,fit_data = cal_Tphi_n.Cal_T2_by_Sw(N)
    Cal_T2_by_Sw_list.append([P_N,Interg])
    plt.plot(t_list/1e-6,P_N,'--',label='N='+str(N)+rf': $T_2^{{1/e}} = {round(T2_e/1e-6,2)} \mu$s')
    
    endtime_i=int(time.time())
    print('N = '+str(N)+' running time: '+str(endtime_i-starttime_i)+' s.')

plt.ylabel(r'$e^{-\langle\chi_N(t)\rangle}$')
plt.xlabel(r'T /$\mu s$')
plt.legend()
plt.show()
Omega_list=np.linspace(0e6,1e6,1000)*2*np.pi

fig, ax1 = plt.subplots()
ax1.plot(Omega_list/1e6/2/np.pi,[S_th(o,kappa,T_eff,g,omega_r,omega_q) for o in Omega_list],
          label=r'$S_q^{th}(\omega)$')
# ax1.plot(Omega_list/1e6/2/np.pi,[S_th(o,kappa,T_eff,g,omega_r,omega_q) for o in Omega_list])
ax1.set_ylabel(r'$S_q^{th}(\omega)$   $Hz$')
ax1.set_xlabel(r'$\omega/2\pi$ MHz')
ax1.legend(loc='upper left')
# ax1 = ax1.twinx()
# ax1.plot(Omega_list/1e6/2/np.pi,[cal_Tphis[0].filter_CPMG(o,10e-6) for o in Omega_list],
#           'r--', label=rf'Filter function $g_{{{N_list[0]}}}$')
# ax1.plot(Omega_list/1e6/2/np.pi,[cal_Tphis[1].filter_CPMG(o,0.00001e-6)*(0.00001e-6)**2 for o in Omega_list],
#           'c--', label=rf'Filter function $g_{{{N_list[1]}}}$')
# ax1.set_ylabel(r'$g_N (\omega, \tau=10 \mu s)$')
# ax1.legend(loc='upper right')

# ax2 = ax1.twinx()
# ax2.plot(Omega_list/1e6/2/np.pi,[cal_Tphis[2].filter_CPMG(o,1e-6) for o in Omega_list],
#           'r--', label=rf'Filter function $g_{{{N_list[2]}}}$')
# ax2.plot(Omega_list/1e6/2/np.pi,[cal_Tphis[3].filter_CPMG(o,1e-6) for o in Omega_list],
#           'c--', label=rf'Filter function $g_{{{N_list[3]}}}$')
# ax2.set_ylabel(r'$g_N (\omega, \tau=10 \mu s)$')
# ax2.legend(loc='upper right')
#%%

"""定义热光场态产生的噪声谱线"""
def S_th(omega_noise,A,
         alpha = 1,
         B = 0):
    omega = omega_noise**alpha
    s_f=A/omega+B
    return(s_f)




kappa=10e6*2*np.pi
T_eff=48e-3
g=120e6*2*np.pi
omega_r=6.5e9*2*np.pi
omega_q=4.5e9*2*np.pi
# tau=20

A=2e11
B=0
alpha=1
N_list=[0]

t_list=np.linspace(0.1e-6,2e-6,40)
cal_Tphis=[]
Cal_T2_by_Sw_list=[]

plt.figure()
for i in range(len(N_list)):
    N=N_list[i]   
    starttime_i=int(time.time())
    
    cal_Tphi_n = CPMG_Tphi(
                  S_w = S_th,
                  N = N,
                  tau_list = t_list,
                  omega_s = 100*2*np.pi,
                  omega_e = 20e6*2*np.pi,
                  int_limit = 5e3, 
                   t_gate = 0e-9,
                  args = {'Sw_args': [A,alpha,B]}, 
                  )
    cal_Tphis.append(cal_Tphi_n)

    P_N,Interg,T2_e = cal_Tphi_n.Cal_T2_by_Sw(N)
    Cal_T2_by_Sw_list.append([P_N,Interg])
    plt.plot(t_list/1e-6,P_N,'--',label='N='+str(N)+rf': $T_2^{{1/e}} = {round(T2_e/1e-6,2)} \mu$s')
    
    endtime_i=int(time.time())
    print('N = '+str(N)+' running time: '+str(endtime_i-starttime_i)+' s.')

plt.ylabel(r'$e^{-\langle\chi_N(t)\rangle}$')
plt.xlabel(r'T /$\mu s$')
plt.legend()
plt.show()
Omega_list=np.linspace(0.1,10e6,1000)*2*np.pi

# fig, ax1 = plt.subplots()
# plt.figure()
# ax1.plot(Omega_list/1e6/2/np.pi,[S_th(o,kappa,T_eff,g,omega_r,omega_q) for o in Omega_list],
#           label=r'$S_q^{th}(\omega)$')
# # ax1.plot(Omega_list/1e6/2/np.pi,[S_th(o,kappa,T_eff,g,omega_r,omega_q) for o in Omega_list])
# ax1.set_ylabel(r'$S_q^{th}(\omega)$   $Hz$')
# ax1.set_xlabel(r'$\omega/2\pi$ MHz')
# ax1.legend(loc='upper left')
# ax1 = ax1.twinx()
# ax1.plot(Omega_list/1e6/2/np.pi,[cal_Tphis[0].filter_Sw_CPMG(o,0.8e-6) for o in Omega_list],
#           'r--', label=rf'Filter function $g_{{{N_list[0]}}}$')
# # ax1.plot(Omega_list/1e6/2/np.pi,[cal_Tphis[1].filter_CPMG(o,0.e-6)*(0.00001e-6)**2 for o in Omega_list],
# #           'c--', label=rf'Filter function $g_{{{N_list[1]}}}$')
# ax1.set_ylabel(r'$g_N (\omega, \tau=10 \mu s)$')
# ax1.legend(loc='upper right')

# ax2 = ax1.twinx()
# ax2.plot(Omega_list/1e6/2/np.pi,[cal_Tphis[2].filter_CPMG(o,1e-6) for o in Omega_list],
#           'r--', label=rf'Filter function $g_{{{N_list[2]}}}$')
# ax2.plot(Omega_list/1e6/2/np.pi,[cal_Tphis[3].filter_CPMG(o,1e-6) for o in Omega_list],
#           'c--', label=rf'Filter function $g_{{{N_list[3]}}}$')
# ax2.set_ylabel(r'$g_N (\omega, \tau=10 \mu s)$')
# ax2.legend(loc='upper right')