# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 13:14:58 2022

@author: mantoutou
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
import time
from joblib import Parallel, delayed
import random
from scipy.fftpack import fft
from scipy.interpolate import interp1d
from scipy import optimize
#%% 绘制过滤函数及积分
#以秒为单位
class CPMG_Tphi:
    def __init__(self, 
              S_w, #比特频率的噪声谱密度函数，单位 omega^2/omega，
              N,   #CPMG脉冲个数，0：Ramsey；1：Spinecho，
              tau_list,  #时间窗口，
              omega_s = None, #积分计算噪声谱密度的起始频率，
              omega_e = None, #积分计算噪声谱密度的结束频率，
              int_limit = 1e4, #频率积分点数，
              t_gate = 0e-9, #pi门时间，
              args = None, #需要具体情况输入的一些额外参数，比如谱密度函数的一些参数，
              inter_point = 1000, #利用插值计算T2的插值点数，
              fit_T2_guess = None, #拟合T2的初猜，
              ):
        self.S_w = S_w
        self.N = int(N)
        self.tau_list = tau_list
        self.omega_s = omega_s
        self.omega_e = omega_e
        self.int_limit = int(int_limit)
        self.t_gate = t_gate
        self.args = args
        self.inter_point = inter_point
        self.fit_T2_guess = fit_T2_guess
        
        
    """cpmg 滤波函数, omega_noise 单位HZ；tau 单位s"""
    def filter_CPMG(self,omega_noise,tau):
        t_gate = self.t_gate
        # print(t_gate)
        # f_noise = omega_noise/2*np.pi
        N = self.N
        if N==0:
            """N=0对应Ramsey时的滤波函数"""
            # print(tau,omega_noise)
            g_N=np.sin(omega_noise*tau/2)**2/(omega_noise*tau/2)**2
        else:
            """CPMG pi门的归一化坐标"""
            normal_position=[1/(2*N)+i/(N) for i in range(N)]
            # normal_position=[np.sin(np.pi*(i+1)/(2*N+2))**2 for i in range(N)]
            g_pi=sum([(-1)**(i+1)*2*np.exp(1j*omega_noise*normal_position[i]*tau)*np.cos(omega_noise*t_gate/2)
                  for i in range(N)])
            g_N=(abs(1+(-1)**(1+N)*np.exp(1j*omega_noise*tau)+g_pi)/(omega_noise*tau))**2
        return (g_N)
    
    """滤波函数*噪声谱, omega_noise 单位HZ；tau 单位s"""
    def filter_Sw_CPMG(self, omega_noise,tau):
        g_N = self.filter_CPMG(omega_noise,tau)
        # print(self.S_w(omega_noise,*self.args['Sw_args']))
        if isinstance(self.args, dict) and 'Sw_args' in self.args:
            f_s = g_N*self.S_w(omega_noise,*self.args['Sw_args'])
        else:
            f_s = g_N*self.S_w(omega_noise)
        return (f_s)
    
    """拟合T2降到1/e时的位置 (同时拟合高斯和指数)"""
    def fit_Tp_func(self,t,T_phi,T_w): 
        return( np.exp(-t**2/T_phi**2-t/T_w))
        
    """利用噪声谱和过滤函数计算T2 """
    def Cal_T2_by_Sw(self,N):
        P_N=[]
        Interg=[]
        tau_list = self.tau_list
        for i in range(len(tau_list)): 
            
            tau=tau_list[i]
            
            """CPMG滤波函数的中心频率"""
            omega_0=1/(2*tau/N)*2*np.pi if N else 0
            omega_s = self.omega_s #if N == 0 else omega_0*0.01
            omega_e = self.omega_e #if N == 0 else omega_0*100
            # print(omega_s,omega_e,N)
            
            
            """噪声谱分部积分 避免跨度太大积分报错 (每隔两个数量级进行积分)"""
            NMin = np.floor(np.log10(omega_s/2/np.pi))
            NMax = np.ceil(np.log10(omega_e/2/np.pi))
            NList = np.arange(NMin,NMax,2)
            
            Inerg_omega_nod = [10**n*2*np.pi for n in NList]
            Inerg_omega_nod[0] = omega_s
            Inerg_omega_nod[-1] = omega_e
            # print(Inerg_omega_nod)
            Int_S = 0
            for ii in range(len(Inerg_omega_nod)-1):
                Int_S_ii, err_ii = integrate.quad(
                    self.filter_Sw_CPMG, Inerg_omega_nod[ii],Inerg_omega_nod[ii+1], 
                    limit=self.int_limit,args = (tau),
                    # epsabs = 1e-24,
                    # epsrel = 1e-24,
                    )
                Int_S += Int_S_ii
            # print(Int_S)
            Int_D = Int_S
            Interg.append(Int_D)
            p=np.exp(-tau**2*Int_D/2)
            P_N.append(p)
        
        """利用拟合P下降到1/e时的T2"""
        if self.fit_T2_guess is None:
            fit_T2_guess = [1*tau_list[np.argmin(abs(np.array(P_N)-1/np.e))],
                            2*tau_list[np.argmin(abs(np.array(P_N)-1/np.e))]]
        else:
            fit_T2_guess = self.fit_T2_guess
        print(fit_T2_guess)
        fit_datas,fit_errors = optimize.curve_fit(self.fit_Tp_func, tau_list, P_N,
                                                  fit_T2_guess,maxfev = 40000)
        a = (1/fit_datas[0])**2; b = 1/fit_datas[1]; c=-1
        T2_e = (-b+np.sqrt(b**2-4*a*c))/(2*a)
        # P_func = interp1d(tau_list, P_N, kind='cubic')
        # T_inter = np.linspace(tau_list[0],tau_list[-1],self.inter_point)
        # P_inter = P_func(T_inter)
        # T2_e = T_inter[(abs(P_inter-1/np.e)).argmin()]
        return(P_N,Interg,T2_e,fit_datas)
def Svv_func(R,omega,T):
    h=6.626e-34
    h_bar=h/2/np.pi
    k_b=1.38065e-23
    
    beta = h_bar*omega/k_b/T
    s_vv = 4*k_b*T*R*(beta/(np.exp(beta)-1))
    
    return(10*np.log10(s_vv))

class Noise_Teff:
    def __init__(self,
            AT_RT,      #室温层衰减器 dB
            AT_50K,     #50K层衰减器 dB
            AT_4K,      #4K层衰减器 dB
            AT_Still,   #Still层衰减器 dB
            AT_CP,     #100mK层衰减器 dB
            AT_MC,      #MC层衰减器 dB
            T_Singal = 300,  #输入信号温度
            T_RT = 300,  #室温层温度
            T_50K = 55,   #50K层温度
            T_4K = 4,    #4K层温度
            T_Still = 0.8,   #Still层温度
            T_CP = 0.1,    #100mK层温度
            T_MC = 0.015,   #MC层温度
            R = 50,
            ):
        self.AT_RT = AT_RT
        self.AT_50K = AT_50K
        self.AT_4K = AT_4K
        self.AT_Still = AT_Still
        self.AT_CP = AT_CP
        self.AT_MC = AT_MC
        self.T_Singal = T_Singal
        self.T_RT = T_RT
        self.T_50K = T_50K
        self.T_4K = T_4K
        self.T_Still = T_Still
        self.T_CP = T_CP
        self.T_MC = T_MC
        self.R = R
    """定义电压噪声谱密度"""
    def Svv_func(self,omega,T):
        h=6.626e-34
        h_bar=h/2/np.pi
        k_b=1.38065e-23
        beta = h_bar*omega/k_b/T
        s_vv = 4*k_b*T*self.R*(beta/(np.exp(beta)-1))
        return(s_vv)
    """叠加每一层的衰减*电压噪声谱密度"""
    def Svv_all_func(self, omega):
        h=6.626e-34
        h_bar=h/2/np.pi
        k_b=1.38065e-23
        AT_list = [self.AT_RT,self.AT_50K,self.AT_4K,
                   self.AT_Still,self.AT_CP,self.AT_MC]
        T_list = [self.T_RT,self.T_50K,self.T_4K,
                  self.T_Still,self.T_CP,self.T_MC]
        S_vv = self.Svv_func(omega,self.T_Singal)
        S_vv_allP = []
        T_eff_P = []
        for p in range(len(AT_list)):
            AT_p = AT_list[p]
            T_p = T_list[p]
            s_p = self.Svv_func(omega,T_p)
            S_vv =(1-1/(10**(AT_p/10)))*s_p+S_vv/(10**(AT_p/10))
            """计算等效温度"""
            T_eff = h_bar*omega/k_b/np.log(4*h_bar*omega*self.R/S_vv+1)
            S_vv_allP.append(S_vv)
            T_eff_P.append(T_eff)
        """依次返回每一层的电压噪声强度及对应频率的等效温度"""
        return(S_vv_allP,T_eff_P)

    """计算最后一层的电压噪声谱密度"""
    def Svv_f_func(self, omega):
        h=6.626e-34
        h_bar=h/2/np.pi
        k_b=1.38065e-23
        AT_list = [self.AT_RT,self.AT_50K,self.AT_4K,
                   self.AT_Still,self.AT_CP,self.AT_MC]
        T_list = [self.T_RT,self.T_50K,self.T_4K,
                  self.T_Still,self.T_CP,self.T_MC]
        S_vv = self.Svv_func(omega,self.T_Singal)
        
        for p in range(len(AT_list)):
            AT_p = AT_list[p]
            T_p = T_list[p]
            s_p = self.Svv_func(omega,T_p)
            S_vv =(1-1/(10**(AT_p/10)))*s_p+S_vv/(10**(AT_p/10))
        """计算MC层等效温度"""
        T_eff = h_bar*omega/k_b/np.log(4*h_bar*omega*self.R/S_vv+1)
        return(S_vv,T_eff)
 #%%
                      
"""cpmg 滤波函数, f_noise 单位HZ"""
def filter_CPMG(f_noise,tau,N,t_gate=40e-9):
    omega_noise=f_noise*2*np.pi
    N=int(N)
    if N==0:
        """N=0对应Ramsey时的滤波函数"""
        g_N=np.sin(omega_noise*tau/2)**2/(omega_noise*tau/2)**2
    else:
        normal_position=[1/(2*N)+i/(N) for i in range(N)]
        g_pi=sum([(-1)**(i+1)*2*np.exp(1j*omega_noise*normal_position[i]*tau)*np.cos(omega_noise*t_gate/2)
              for i in range(N)])
        g_N=(abs(1+(-1)**(1+N)*np.exp(1j*omega_noise*tau)+g_pi)/(omega_noise*tau))**2
    return (g_N)

"""cpmg 滤波函数*1/f噪声谱, omega 单位HZ"""
def filter_1f_CPMG (f_noise,tau,N,alpha,t_gate=65e-9):
    omega_noise=f_noise*2*np.pi
    N=int(N)
    if N==0:
        g_N=np.sin(omega_noise*tau/2)**2/(omega_noise*tau/2)**2/(omega_noise/2/np.pi)**alpha*tau**(1-alpha)
    else:
        normal_position=[1/(2*N)+i/(N) for i in range(N)]
        g_pi=sum([(-1)**(i+1)*2*np.exp(1j*omega_noise*normal_position[i]*tau)*np.cos(omega_noise*t_gate/2) 
              for i in range(N)])
        g_N=(abs(1+(-1)**(1+N)*np.exp(1j*omega_noise*tau)+g_pi)/(omega_noise*tau))**2/(omega_noise/2/np.pi)**alpha*tau**(1-alpha)
    return (g_N)

def Gauss_func(f,a,b,c):
    return(a*np.exp(-(f-b)**2/2/c**2))

def filter_1fwGauss_CPMG (f_noise,tau,N,alpha,t_gate=40e-9):
    omega_noise=f_noise*2*np.pi
    N=int(N)
    if N==0:
        g_N=np.sin(omega_noise*tau/2)**2/(omega_noise*tau/2)**2*(1/(omega_noise/2/np.pi)**alpha+Gauss_func(omega_noise/2/np.pi,6e-3/0.0217*2*(2*np.pi)**2,0.376,10e-3))*tau**(1-alpha)
    else:
        normal_position=[1/(2*N)+i/(N) for i in range(N)]
        g_pi=sum([(-1)**(i+1)*2*np.exp(1j*omega_noise*normal_position[i]*tau)*np.cos(omega_noise*t_gate/2) 
              for i in range(N)])
        g_N=(abs(1+(-1)**(1+N)*np.exp(1j*omega_noise*tau)+g_pi)/(omega_noise*tau))**2*(1/(omega_noise/2/np.pi)**alpha+Gauss_func(omega_noise/2/np.pi,6e-3/0.0217*2*(2*np.pi)**2,0.376,10e-3))*tau**(1-alpha)
    return (g_N)
def fit_S1f(N,A,B,C):
    r=A/N+B/N**2+C
    return(r)
def int_S1f_CPMG(N,S_1f,alpha):
    tau=60e-6
    integ, err = integrate.quad(filter_1f_CPMG, 1, 20e6, limit=500,args = (tau,N,alpha,40e-9))
    return(S_1f*integ)

def int_S1f_Ramsey(omega_off,S_1f,alpha):
    tau=10e-6
    integ, err = integrate.quad(filter_1f_CPMG, omega_off, 40e6, limit=500,args = (tau,0,alpha,40e-9))
    return(S_1f*integ)


#%% delta近似计算噪声强度
#以微秒为单位
def Cal_1f_by_t(tau_list,N,P_list,Tw_a):
    Omega_list=np.linspace(1e-6,8,2001)
    # N_list=[1.0, 2.0, 5.0, 8.0, 12.0, 17.0, 23.0, 30.0, 37.0, 46.0, 55.0, 65.0, 75.0, 87.0, 100.0]
    S_N=[]
    Interg=[]
    F_0=[]
    for i in range(len(tau_list)):
        p=P_list[i]
        tau=tau_list[i]
        fre_0=1/(2*tau/N)
        fre_s=1/(2*tau/N)*0.4
        fre_e=1/(2*tau/N)*2
        # omega_noise= sp.symbols("omega")
        Int, err = integrate.quad(filter_CPMG, fre_s,fre_e, limit=500,args = (tau,N,40e-3))
        Interg.append(Int)
        s=(np.log(p)+tau/Tw_a/2)/(-tau**2/2*Int)
        S_N.append(s)
        F_0.append(1/(2*tau/N))
        # A = sp.integrate
    return(Int,S_N,F_0)

def Cal_Sw_by_t(tau_list,N,P_list):
    Omega_list=np.linspace(1e-6,8,2001)
    # N_list=[1.0, 2.0, 5.0, 8.0, 12.0, 17.0, 23.0, 30.0, 37.0, 46.0, 55.0, 65.0, 75.0, 87.0, 100.0]
    S_N=[]
    Interg=[]
    F_0=[]
    for i in range(len(tau_list)):
        p=P_list[i]
        tau=tau_list[i]
        fre_0=1/(2*tau/N)
        fre_s=1/(2*tau/N)*0.4
        fre_e=1/(2*tau/N)*2
        # omega_noise= sp.symbols("omega")
        Int, err = integrate.quad(filter_CPMG, fre_s,fre_e, limit=500,args = (tau,N,40e-3))
        Interg.append(Int)
        s=np.log(p)/(-tau**2/2*Int)
        S_N.append(s)
        F_0.append(1/(2*tau/N))
        # A = sp.integrate
    return(Int,S_N,F_0)

