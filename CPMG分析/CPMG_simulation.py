# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 12:24:15 2022

@author: mantoutou
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import optimize, interpolate
from CPMG_filter_function import*
from joblib import Parallel, delayed

def smooth(x,num):
    """定义smooth函数"""
    if num // 2 == 0: # 偶数转奇数
        num -= 1
    length = len(x)
    y = np.zeros(length)
    N = (num - 1) / 2
    for i in range(0, length):
        cont_0  = i
        cont_end = length - i - 1
        if cont_0 in range(0,int(N)) or cont_end in range(0,int(N)):
            cont = min(cont_0 ,cont_end)
            y[i] = np.mean(x[i - cont : i + cont + 1])
        else:
            y[i] = np.mean(x[i - int(N) : i + int(N) + 1])
    return y
def Fit_PSD_1f_log_func(f,A,alpha,B):
    return(np.log(A/f**alpha+B))
def noise_data_process(PSD_r,Freq_r):
    """
    将噪声谱数据转化为可用的噪声谱函数
    """
    
    """对原始数据进行smooth平均后再利用插值形成噪声谱函数形式"""

    PSD_r_s = smooth(PSD_r, 5)
    func_inter_log=interpolate.UnivariateSpline(np.log(Freq_r),
                                                [np.log(psd) for psd in PSD_r_s],
                                                k=5,s=31)
    func_inter = lambda f: np.exp(func_inter_log(np.log(f)))
    PSD_inter=func_inter(Freq_r)

    """对低于1HZ的数据进行拟合得到1/f噪声的拟合结果"""
    index_1f = np.where(np.array(Freq_r)<1)[0]
    amp_0 = ((PSD_r_s[index_1f][0] - PSD_r_s[index_1f][-1])/
             (1/Freq_r[index_1f][0] - 1/Freq_r[index_1f][-1]))
    fit_1f_data,fit_1f_error = optimize.curve_fit(Fit_PSD_1f_log_func,Freq_r[index_1f],
                                            [np.log(psd) for psd in PSD_r_s[index_1f]], 
                                            p0 = [amp_0,1,PSD_r_s[index_1f][-1]], 
                                            # bounds = (0,np.inf),
                                            maxfev = 400000)
    PSD_1f_fit = np.exp(Fit_PSD_1f_log_func(Freq_r,*fit_1f_data))

    """对高于10MHZ的数据进行平均得到白噪声结果"""
    index_w = np.where(np.array(Freq_r)>10e6)[0]
    PSD_w = np.mean(PSD_r[index_w])

    """
    拟合数据和白噪声转折点，
    寻找符合区间内拟合值(或平均得到的白噪值)与插值数据最靠近的点作为转折点，
    可以避免曲线出现突然的变化
    """
    transf_1f = np.argmin(np.abs(np.log10(PSD_inter[index_1f])-np.log10(PSD_1f_fit[index_1f])))
    transf_w = np.argmin(np.abs(np.log10(PSD_inter)-np.log10(PSD_w)))
    
    return(func_inter,fit_1f_data, PSD_w, transf_1f, transf_w)

def Gauss_Noise_func(omega,S_g,omega_g,sigma_g):
    f = omega/2/np.pi
    f_g = omega_g/2/np.pi
    sigma_g = sigma_g/2/np.pi
    return(S_g*np.exp(-(f-f_g)**2/2/sigma_g**2))
def f1w_Noise_fuc(omega,S_1f,S_w,alpha):
    f = omega/2/np.pi
    return(S_1f*f**-alpha+S_w)
#%%  Foe Example
starttime=int(time.time())

"""生成模拟噪声谱数据进行演示"""
S_1f = 1.5e-11
S_w = 1.4e-16
alpha = 1.1
Freq_r = np.linspace(1e-3**0.1, 50e6**0.1,5001)
Freq_r = Freq_r**10

S_g1=8e-13
omega_g1=0.3e6*2*np.pi
sigma_g1=50e3

S_g2=0.7e-14
omega_g2=-0.3e6*2*np.pi
sigma_g2=20e5


PSD_func = lambda omega: (f1w_Noise_fuc(omega,S_1f,S_w,alpha)+
                          Gauss_Noise_func(omega,S_g1,omega_g1,sigma_g1)+
                          Gauss_Noise_func(omega,S_g2,omega_g2,sigma_g2)
                          )

PSD_Svv = [PSD_func(f*2*np.pi) for f in Freq_r]
PSD_dBm = [10*np.log10(P_s/1e-3/(4*50)) for P_s in PSD_Svv]


fig,axes=plt.subplots()
twin_axes=axes.twinx() 
# axes.plot(Freq_r,PSD_r, label = 'AWG PSD')
axes.plot(Freq_r,PSD_Svv, label = 'AWG PSD')

axes.set_xscale('log')
axes.set_yscale('log')

#%%
"""
输入室温噪声谱的频率和谱密度(dBm)，
要求单边噪声谱，频率范围最小要1mHz， 最大要超过50MHz
也可以输入芯片前端噪声谱数据，此时需将参数 is_psd_in_chip = True
"""
kv_c_16M = 38260057580.079124  #partial(f)/partial(V)  coupler在16M耦合处的频率随电压变化的斜率
kv_c_off = 20148689377.66139  #partial(f)/partial(V)  coupler在关断点处的频率随电压变化的斜率
leakage_C16M_ZCZ3 = 0.0216                #ZCZ3 比特在打开16M耦合处的在coupler上的态泄露
leakage_Coff_ZCZ3 = 0.00615                    #ZCZ3 比特在关断点的在coupler上的态泄露
kv_q = 2214117455.179478*1.5

def PSD_omega(omega):
    # psd_16M = PSD_func(omega/2/np.pi)*kv_c_16M**2*(2*np.pi)**2*leakage_C16M_ZCZ3
    psd_off = PSD_func(omega)*kv_q**2*(2*np.pi)**2
    psd = psd_off
    return psd/2/np.pi


N_list=[0]

t_list=np.linspace(0.02e-6,5e-6,201)
cal_Tphis=[]
Cal_T2_by_Sw_list=[]

plt.figure()
for i in range(len(N_list)):
    N=N_list[i]   
    starttime_i=int(time.time())
    
    cal_Tphi_n = CPMG_Tphi(
                  S_w = PSD_omega,
                  N = N,
                  tau_list = t_list,
                  omega_s = 0.005*2*np.pi,
                  omega_e = 50e6*2*np.pi,
                  int_limit = 5e3, 
                  t_gate = 0e-9, )
    cal_Tphis.append(cal_Tphi_n)
    P_N,Interg,T2_e ,fit_datas= cal_Tphi_n.Cal_T2_by_Sw(N)
    Cal_T2_by_Sw_list.append([P_N,Interg])
    plt.plot(t_list/1e-6,P_N,'--',label='N='+str(N)+' simulated modulus')
    # plt.plot(t_list/1e-6,cal_Tphi_n.fit_Tp_func(t_list,*fit_datas),
    #          label='N='+str(N)+rf': fited $T_2^{{1/e}} = {round(T2_e/1e-6,2)} \mu$s')
    endtime_i=int(time.time())
    print('N = '+str(N)+' running time: '+str(endtime_i-starttime_i)+' s.')
plt.ylabel(r'$e^{-\langle\chi_N(t)\rangle}$')
plt.xlabel(r'T /$\mu s$')
plt.legend()
plt.show()               

Omega_list=np.linspace(0.1,2e6,1000)*2*np.pi

fig,axes=plt.subplots()
twin_axes=axes.twinx() 
axes.plot(Freq_r,PSD_Svv, label = 'AWG PSD')

axes.set_xscale('log')
axes.set_yscale('log')
y1, y2 = axes.get_ylim() 
axes.set_ylabel(r'$S_{vv}~ (V^2/Hz)$')
axes.set_xlabel('Freq (HZ)')
twin_axes.plot(Omega_list/2/np.pi,[cal_Tphis[0].filter_CPMG(o, 1e-6) for o in Omega_list],
          'r--', label=rf'Filter function $g_{{{N_list[0]}}}$')
twin_axes.plot(Omega_list/2/np.pi,[cal_Tphis[0].filter_CPMG(o, 4e-6) for o in Omega_list],
          'b--', label=rf'Filter function $g_{{{N_list[0]}}}$')
twin_axes.set_ylabel(r'$S_q^{th}(\omega)$   $Hz$')
twin_axes.set_xlabel(r'$\omega/2\pi$ MHz')

axes.legend()
twin_axes.legend(loc='upper left')
plt.show()

endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')
#%%

starttime=int(time.time())

def Gauss_func(f,a,b,c):
    return(a*np.exp(-(f-b)**2/2/c**2))
def f1_fuc(f,S_0,alpha):
    return(S_0*f**-alpha)

def S_noise(f,a,b,c,S_0,alpha):
    return(Gauss_func(f,a,b,c)+f1_fuc(f,S_0,alpha))

def filter_1fwGauss_CPMG (omega_noise,tau,N,alpha,a_1,b_1,c_1,a_2,b_2,c_2,t_gate=60e-3):
    omega_noise=omega_noise*2*np.pi
    N=int(N)
    if N==0:
        g_N=np.sin(omega_noise*tau/2)**2/(omega_noise*tau/2)**2*(1/(omega_noise/2/np.pi)**alpha+Gauss_func(omega_noise/2/np.pi,a_1,b_1,c_1)+Gauss_func(omega_noise/2/np.pi,a_2,b_2,c_2))*tau**(1-alpha)
    else:
        normal_position=[1/(2*N)+i/(N) for i in range(N)]
        g_pi=sum([(-1)**(i+1)*2*np.exp(1j*omega_noise*normal_position[i]*tau)*np.cos(omega_noise*t_gate/2) 
              for i in range(N)])
        g_N=(abs(1+(-1)**(1+N)*np.exp(1j*omega_noise*tau)+g_pi)/(omega_noise*tau))**2*(1/(omega_noise/2/np.pi)**alpha+Gauss_func(omega_noise/2/np.pi,a_1,b_1,c_1)+Gauss_func(omega_noise/2/np.pi,a_2,b_2,c_2))*tau**(1-alpha)
    return (g_N)

f_range=np.linspace(0.01,2,1000)
a_1=100
b_1=0.053
c_1=60e-3
a_2=10
b_2=0.787
c_2=10e-3
S_0=0.0117
N_num=15
N_list=np.linspace(1**0.5,100**0.5,N_num)**2
N_list=np.array(N_list,dtype='i8')
N_list=[0]
a_list=[10,20,40,80]
Time_r=np.linspace(0.01,50,180)
# for k in range(len(a_list)):
def multijob_CPMG_simulation(k):
    # a=a_list[k]
    N_i=N_list[k]
    Gamma_phi=[-tau**1.9*(S_0*integrate.quad(filter_1fwGauss_CPMG, 1.5e-2,2, limit=10000,args = (tau,N_i,0.9,a_1,b_1,c_1,a_2,b_2,c_2,60e-3))[0])/2-tau/100 for tau in Time_r] 
    # CPMG_simu.append([np.exp(i) for i in Gamma_phi])
    return([np.exp(i) for i in Gamma_phi])
data=Parallel(n_jobs=8, verbose=2)(delayed(multijob_CPMG_simulation)(k) for k in range(len(N_list)))
CPMG_simu=np.array(data)
# plt.figure()
# X_m=[-(Time_r[0]+Time_r[1])/2]+[(Time_r[i+1]+Time_r[i])/2 for i in range(len(Time_r)-1)]+[Time_r[-1]+(Time_r[-1]-Time_r[-2])/2]
# Y_m=[-(N_list[0]+N_list[1])/2]+[(N_list[i+1]+N_list[i])/2 for i in range(len(N_list)-1)]+[N_list[-1]+(N_list[-1]-N_list[-2])/2]
# # X,Y=np.meshgrid(Time_r,N_list)
# X,Y=np.meshgrid(X_m,Y_m)
# plt.pcolor(Y,X,CPMG_simu,cmap='jet')
# plt.colorbar()
# plt.ylabel(r'time $\mu s$')
# plt.xlabel(r'Num of $X_\pi$')

plt.figure()
for ii in range(len(N_list)):
    plt.plot(Time_r,CPMG_simu[ii])

# plt.figure()
# plt.plot(f_range,[S_noise(f,a_1*S_0,b_1,c_1,S_0,0.9) for f in f_range])
tau = 1
plt.figure()
plt.plot(f_range,
        [filter_1fwGauss_CPMG (f,1,0,1,a_1,b_1,c_1,a_2,b_2,c_2,t_gate=60e-3)
         for f in f_range])
plt.plot(f_range,
        [filter_1fwGauss_CPMG (f,2,0,1,a_1,b_1,c_1,a_2,b_2,c_2,t_gate=60e-3)
         for f in f_range])
# plt.xscale('log',)
# plt.yscale('log',)
# plt.plot(f_range)


endtime=int(time.time())
print('total run time: ',endtime-starttime,'s')
#%%

starttime=int(time.time())

def Gauss_func(f,a,b,c):
    return(a*np.exp(-(f-b)**2/2/c**2))
def f1_fuc(f,S_0,alpha):
    return(S_0*f**-alpha)

def S_noise(f,a,b,c,S_0,alpha):
    return(Gauss_func(f,a,b,c)+f1_fuc(f,S_0,alpha))

def filter_1fwGauss_CPMG (omega_noise,tau,N,alpha,a_1,b_1,c_1,a_2,b_2,c_2,t_gate=60e-3):
    omega_noise=omega_noise*2*np.pi
    N=int(N)
    if N==0:
        g_N=np.sin(omega_noise*tau/2)**2/(omega_noise*tau/2)**2*(1/(omega_noise/2/np.pi)**alpha+Gauss_func(omega_noise/2/np.pi,a_1,b_1,c_1)+Gauss_func(omega_noise/2/np.pi,a_2,b_2,c_2))*tau**(1-alpha)
    else:
        normal_position=[1/(2*N)+i/(N) for i in range(N)]
        g_pi=sum([(-1)**(i+1)*2*np.exp(1j*omega_noise*normal_position[i]*tau)*np.cos(omega_noise*t_gate/2) 
              for i in range(N)])
        g_N=(abs(1+(-1)**(1+N)*np.exp(1j*omega_noise*tau)+g_pi)/(omega_noise*tau))**2*(1/(omega_noise/2/np.pi)**alpha+Gauss_func(omega_noise/2/np.pi,a_1,b_1,c_1)+Gauss_func(omega_noise/2/np.pi,a_2,b_2,c_2))*tau**(1-alpha)
    return (g_N)

f_range=np.linspace(0.01,2,100)
a_1=20
b_1=0.373
c_1=10e-3
a_2=10
b_2=0.787
c_2=10e-3
S_0=0.0117
N_num=15
N_list=np.linspace(1**0.5,100**0.5,N_num)**2
N_list=np.array(N_list,dtype='i8')
a_list=[10,20,40,80]
Time_r=np.linspace(0.01,90,180)
# for k in range(len(a_list)):
def multijob_CPMG_simulation(k):
    # a=a_list[k]
    N_i=N_list[k]
    Gamma_phi=[-tau**1.9*(S_0*integrate.quad(filter_1fwGauss_CPMG, 1.5e-4,40, limit=1000,args = (tau,N_i,0.9,a_1,b_1,c_1,a_2,b_2,c_2,60e-3))[0])/2-tau/100 for tau in Time_r] 
    # CPMG_simu.append([np.exp(i) for i in Gamma_phi])
    return([np.exp(i) for i in Gamma_phi])
data=Parallel(n_jobs=8, verbose=2)(delayed(multijob_CPMG_simulation)(k) for k in range(len(N_list)))
CPMG_simu=np.array(data)
plt.figure()
X_m=[-(Time_r[0]+Time_r[1])/2]+[(Time_r[i+1]+Time_r[i])/2 for i in range(len(Time_r)-1)]+[Time_r[-1]+(Time_r[-1]-Time_r[-2])/2]
Y_m=[-(N_list[0]+N_list[1])/2]+[(N_list[i+1]+N_list[i])/2 for i in range(len(N_list)-1)]+[N_list[-1]+(N_list[-1]-N_list[-2])/2]
# X,Y=np.meshgrid(Time_r,N_list)
X,Y=np.meshgrid(X_m,Y_m)
plt.pcolor(Y,X,CPMG_simu,cmap='jet')
plt.colorbar()
plt.ylabel(r'time $\mu s$')
plt.xlabel(r'Num of $X_\pi$')

# plt.figure()
# for ii in range(len(a_list)):
#     plt.plot(Time_r,CPMG_simu[ii])

plt.figure()
plt.plot(f_range,[S_noise(f,a_1*S_0,b_1,c_1,S_0,0.9) for f in f_range])

endtime=int(time.time())
print('total run time: ',endtime-starttime,'s')