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

def Teff_func(P_s, # 噪声谱功率 dBm, P_s = V_n*I_n
              B_w,  # 采样带宽 Hz
              ):
    k_b= 1.3806505e-23
    kTB = 10**(P_s/10)*1e-3
    T_eff = kTB/(B_w*k_b)
    return(T_eff)
    
folder_name = r'C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\读取光子数噪声\ReadIn_AWG_频谱仪'

file_MWS_00dBm = folder_name + r'\Trace_-00dBm_MWS.csv'
file_MWS_30dBm = folder_name + r'\Trace_-30dBm_MWS.csv'
file_MWS_50dBm = folder_name + r'\Trace_-50dBm_MWS.csv'
file_Awg_MWS_20dBm = folder_name + r'\Trace_Awg_MWS-20dBm.csv'
file_Awg_MWS_off = folder_name + r'\Trace_Awg_MWS-off.csv'
file_50Ohm = folder_name + r'\Trace_50Ohm_Base.csv'

file_AWG20dBm_woAMP_BW5M =  folder_name + r'\Trace_AWG20dBm_woAMP_BW5M.csv'
file_50Ohm_woAMP_BW5M =  folder_name + r'\Trace_50Ohm_woAMP_BW5M.csv'
file_AWG20dBm_woAMP =  folder_name + r'\Trace_AWG20dBm_woAMP.csv'
file_50Ohm_woAMP =  folder_name + r'\Trace_50Ohm_woAMP.csv'

SA_MWS_00dBm = list(map(list,zip(*Get_csv_data(file_MWS_00dBm,45,-1))))
SA_MWS_30dBm = list(map(list,zip(*Get_csv_data(file_MWS_30dBm,45,-1))))
SA_MWS_50dBm = list(map(list,zip(*Get_csv_data(file_MWS_50dBm,45,-1))))
SA_Awg_MWS_20dBm = list(map(list,zip(*Get_csv_data(file_Awg_MWS_20dBm,45,-1))))
SA_Awg_MWS_off = list(map(list,zip(*Get_csv_data(file_Awg_MWS_off,45,-1))))
SA_50Ohm = list(map(list,zip(*Get_csv_data(file_50Ohm,45,-1))))

SA_AWG20dBm_woAMP_BW5M = list(map(list,zip(*Get_csv_data(file_AWG20dBm_woAMP_BW5M,45,-1))))
SA_50Ohm_woAMP_BW5M = list(map(list,zip(*Get_csv_data(file_50Ohm_woAMP_BW5M,45,-1))))
SA_AWG20dBm_woAMP = list(map(list,zip(*Get_csv_data(file_AWG20dBm_woAMP,45,-1))))
SA_50Ohm_woAMP = list(map(list,zip(*Get_csv_data(file_50Ohm_woAMP,45,-1))))

# label_list1 = [ '50 Ohm load','MWS: -30 dBm', 'MWS: -50 dBm', 'MWS: 0 dBm']
# plt.figure()
# plt.plot(SA_50Ohm[0],SA_50Ohm[1],label = '50 Ohm load')
# plt.plot(SA_MWS_30dBm[0],SA_MWS_30dBm[1], label = 'MWS Power: -30 dBm')
# plt.plot(SA_MWS_50dBm[0],SA_MWS_50dBm[1],label = 'MWS Power: -50 dBm')
# plt.plot(SA_MWS_00dBm[0],SA_MWS_00dBm[1],label = 'MWS Power: 0 dBm')
# plt.ylabel('PSD with BW = 1 MHz (dB)')
# plt.xlabel('Freq (HZ)')
# plt.legend()

# B_w = 1e6
# plt.figure()
# plt.plot(SA_50Ohm[0],[Teff_func(p,B_w) for p in SA_50Ohm[1]],
#          label = '50 Ohm load')
# plt.plot(SA_Awg_MWS_20dBm[0],[Teff_func(p,B_w) for p in SA_Awg_MWS_20dBm[1]], 
#          label = 'AWG Noise with LO = 20 dBm')
# plt.plot(SA_Awg_MWS_off[0],[Teff_func(p,B_w) for p in SA_Awg_MWS_off[1]], 
#          label = 'AWG Noise with LO off')

# plt.yscale('log')
# plt.ylabel(r'Noise $T_{eff}$ (K)')
# plt.xlabel('Freq (HZ)')
# plt.legend()

"""
存在前置放大时，AWG噪声与50本底噪声对比
同曲线双y轴绘图，左边显示噪声谱，右边显示等效温度
"""
fig,axes=plt.subplots()
twin_axes=axes.twinx() 
axes.plot(SA_50Ohm[0],[p - 10*(np.log10(1e6)) for p in SA_50Ohm[1]],
          label = '50 Ohm load')
axes.plot(SA_Awg_MWS_20dBm[0],[p - 10*(np.log10(1e6)) for p in SA_Awg_MWS_20dBm[1]], 
         label = 'AWG Noise with LO = 20 dBm')
axes.plot(SA_Awg_MWS_off[0],[p - 10*(np.log10(1e6)) for p in SA_Awg_MWS_off[1]], 
         label = 'AWG Noise with LO off')
# plt.plot(SA_Awg_MWS_off[0],SA_Awg_MWS_off[1], label = 'AWG PSD with LO off')
y1, y2 = axes.get_ylim() 
axes.set_ylabel('PSD (dBm/Hz)')
axes.set_xlabel('Freq (HZ)')
twin_axes.set_ylim(Teff_func(y1,1),Teff_func(y2,1))
twin_axes.set_ylabel(r'Noise $T_{eff}$ (K)')
twin_axes.set_yscale('log')
axes.legend()
plt.show()


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



