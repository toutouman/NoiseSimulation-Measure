# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 19:14:18 2023

@author: mantoutou
"""

import numpy as np
import matplotlib.pyplot as plt
import csv


""" 定义数据提取函数 """
def Get_csv_data(file_name, S_index, E_index):
    data_list = []
    with open(file_name, 'r', encoding = 'utf-8') as f_l:
        for row in csv.reader(f_l):
            data_list.append(row)
        # read_date = f_l.readline()
    data_return = list(map(lambda x: list(map(float,x)), data_list[S_index:E_index]))
    return(data_return)

""" 定义将频谱仪数据转化为等效温度函数 """
def Teff_func(P_s, # 噪声谱功率 dBm, P_s = V_n*I_n
              B_w,  # 采样带宽 Hz
              ):
    k_b= 1.3806505e-23
    kTB = 10**(P_s/10)*1e-3
    T_eff = kTB/(B_w*k_b)
    return(T_eff)

""" 定义将频谱仪数据转化为S_vv函数 """
def Svv_cal_func(P_s, # 噪声谱功率 dBm, P_s = V_n*I_n
                 B_w,  # 采样带宽 Hz
              ):
    S_vv = (10**(P_s/10))*1e-3/B_w*4*50
    return(S_vv)

folder_name = r'C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\Z板子噪声室温测试\频谱仪测试数据'


file_ZPA = folder_name + r'\Trace_ZPA.csv'
file_50Ohm = folder_name + r'\Trace_50Ohm.csv'



SA_Awg_ZPA = list(map(list,zip(*Get_csv_data(file_ZPA,45,-1))))
SA_50Ohm = list(map(list,zip(*Get_csv_data(file_50Ohm,45,-1))))


"""
存在前置放大时，AWG噪声与50本底噪声对比
同曲线双y轴绘图，左边显示噪声谱，右边显示S_vv
"""
fig,axes=plt.subplots()
twin_axes=axes.twinx() 
axes.plot(SA_50Ohm[0],[p - 10*(np.log10(5.1e3)) for p in SA_50Ohm[1]],
          label = '50 Ohm load')
axes.plot(SA_Awg_ZPA[0],[p - 10*(np.log10(5.1e3)) for p in SA_Awg_ZPA[1]], 
         label = 'AWG Noise ')

# plt.plot(SA_Awg_MWS_off[0],SA_Awg_MWS_off[1], label = 'AWG PSD with LO off')
axes.set_xscale('log')
y1, y2 = axes.get_ylim() 
axes.set_ylabel('PSD (dBm/Hz)')
axes.set_xlabel('Freq (HZ)')
twin_axes.set_ylim(Svv_cal_func(y1,1),Svv_cal_func(y2,1))
twin_axes.set_ylabel(r'$S_{vv}~ (V^2/Hz)$')
twin_axes.set_yscale('log')
axes.legend()
plt.show()



AMG_Compare_wAMP = [(Svv_cal_func(p_a,5.1e3)-Svv_cal_func(p_b,5.1e3)) for p_a,p_b in 
                     zip(SA_Awg_ZPA[1],SA_50Ohm[1])]
plt.figure()
plt.plot(SA_Awg_ZPA[0][1:],AMG_Compare_wAMP[1:], 
          label = 'AWG Noise')
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'PSD $S_{vv}~( V^2/Hz)$')
plt.xlabel('Freq (HZ)')
plt.legend()
plt.show()

#%%

# plt.figure()
# plt.plot(SA_50Ohm_woAMP_BW5M[0],SA_50Ohm_woAMP_BW5M[1],label = '50 Ohm load')
# plt.plot(SA_AWG20dBm_woAMP_BW5M[0],SA_AWG20dBm_woAMP_BW5M[1], label = 'AWG PSD with LO = 20 dBm')
# # plt.plot(SA_Awg_MWS_off[0],SA_Awg_MWS_off[1], label = 'AWG PSD with LO off')
# plt.ylabel('PSD with BW = 5 MHz (dB)')
# plt.xlabel('Freq (HZ)')
# plt.legend()


"""
无前置放大时，AWG噪声与50本底噪声对比
同曲线双y轴绘图，左边显示噪声谱，右边显示等效温度
"""
fig,axes=plt.subplots()
axes.plot(SA_50Ohm_woAMP[0],[p - 10*(np.log10(1e6)) for p in SA_50Ohm_woAMP[1]],
          label = '50 Ohm load')
axes.plot(SA_AWG20dBm_woAMP[0],[p - 10*(np.log10(1e6)) for p in SA_AWG20dBm_woAMP[1]],
          label = 'AWG PSD with LO = 20 dBm')

y1, y2 = axes.get_ylim()
axes.set_ylabel('PSD (dBm/Hz)')
axes.set_xlabel('Freq (HZ)')

twin_axes=axes.twinx() 
twin_axes.set_ylim(Teff_func(y1,1),Teff_func(y2,1))
twin_axes.set_ylabel(r'Noise $T_{eff}$ (K)')
twin_axes.set_yscale('log')

axes.legend()
plt.show()



AMG_Tcompare_woAMP = [Teff_func(p_a,1e6)-Teff_func(p_b,1e6) for p_a,p_b in 
                     zip(SA_AWG20dBm_woAMP[1],SA_50Ohm_woAMP[1])]
AMG_Tcompare_woAMP_BW5M = [Teff_func(p_a,5e6)-Teff_func(p_b,5e6) for p_a,p_b in 
                     zip(SA_AWG20dBm_woAMP_BW5M[1],SA_50Ohm_woAMP_BW5M[1])]
AMG_Tcompare_wAMP = [Teff_func(p_a,1e6)-Teff_func(p_b,1e6) for p_a,p_b in 
                     zip(SA_Awg_MWS_20dBm[1],SA_50Ohm[1])]

plt.figure()
plt.plot(SA_50Ohm_woAMP[0],AMG_Tcompare_woAMP, 
          label = 'AWG Noise wo-PreAmplifier')
# plt.plot(SA_AWG20dBm_woAMP_BW5M[0],AMG_Tcompare_woAMP_BW5M, 
#           label = 'AWG Noise wo-Preamplifier (BW = 5 MHZ)')
plt.plot(SA_Awg_MWS_20dBm[0],AMG_Tcompare_wAMP, 
          label = 'AWG Noise w-PreAmplifier')
plt.yscale('log')
plt.ylabel(r'Noise $T_{eff}$ (K)')
plt.xlabel('Freq (HZ)')
plt.legend()



