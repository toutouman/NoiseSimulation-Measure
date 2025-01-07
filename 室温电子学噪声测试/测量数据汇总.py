# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:16:38 2023

@author: mantoutou
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import csv
from scipy import interpolate, optimize
from scipy.signal import savgol_filter
import sys
sys.path.append(r'C:\Users\mantoutou\OneDrive\文档\程序\科大\噪声仿真&测试\CPMG分析')
from CPMG_filter_function import Noise_Teff, CPMG_Tphi

#%% 处理函数定义


"""定义等间隔平均PSD 函数"""
def PSD_Avg_Equal(F_list,
            PSD_list,
            interval_num = 200,  #整段频域分为interval_num个间隔进行平均
            ):
    F_num = len(F_list)
    PSD_avg_list = []
    F_avg_list = []
    for ii in range(interval_num):
        interval = int((F_num)/interval_num)
        index = np.arange(ii*interval,(ii+1)*interval, dtype = 'int')
        F_avg_list.append(np.mean(F_list[index]))
        """返回PSD 单位为 V^2/Hz"""
        PSD_avg_list.append(np.mean(PSD_list[index]))
        # """返回PSD 单位为 dBm/Hz"""
        # PSD_avg_list.append(10*np.log10(np.mean(PSD_list[index])/100/1e-3))
    return (F_avg_list, PSD_avg_list)

"""定义按指数间隔平均FFT 函数"""
def FFT_Avg_Order(F_list,
                  PSD_list,
                  order_base = 2,  #以order_base为底进行指数间隔
                  order_num = 10,  #每个数量级之间再分割若干个间隔进行平均，
                  ):
    """获取最高，最低数量级"""
    O_max = np.ceil(math.log(np.max(F_list),order_base))
    O_min = np.floor(math.log(F_list[1]-F_list[0],order_base)) 
    Order_list = np.arange(O_min,O_max,1)
    
    PSD_avg = []
    F_avg = []
    for ii in Order_list:
        for jj in range(order_num):
            indexs = np.where((np.array(F_list)>2**(ii+jj/order_num)) &
                             (np.array(F_list)<=2**(ii+(jj+1)/order_num)))
            F_avg.append(np.mean(F_list[indexs]))
            """返回PSD 单位为 V^2/Hz"""
            PSD_avg.append(np.mean(PSD_list[indexs]))
            # """返回PSD 单位为 dBm/Hz"""
            # PSD_avg.append(10*np.log10(np.mean(PSD_list[indexs])/100/1e-3))
    return(F_avg, PSD_avg)


"""定义噪声信号 FFT 处理函数"""
def Noise_FFT_Avg(Noise_list,
                  sample_rate,
                  is_2side_psd = False,  #返回噪声谱为单边还是双边噪声谱，选择单边会在正频轴的PSD乘2
                  is_equal_avg = True,  #噪声谱在频域上进行平均时选择等间隔平均还是按指数平均
                  interval_num = 200,   #选择等间隔平均时，整段频域分为interval_num个间隔进行平均
                  order_base = 2,    #选择指数平均时以order_base为底进行指数间隔
                  order_num = 20,    #选择指数平均时，每个数量级之间再分割若干个间隔进行平均，
                  ):
    if is_2side_psd:
        """经过FFT后得到双边噪声谱"""
        Noise_FFT_Lists = [abs(np.fft.fftshift(np.fft.fft(N_data)))**2/int(len(N_data)*sample_rate)
                           for N_data in Noise_list]
        Fre_Lists =[(np.linspace(0,sample_rate,len(N_data))-
                     int(len(N_data)/2)/((len(N_data)-1)/sample_rate))
                    for N_data in Noise_list]
    else:
        """单边噪声谱"""
        Noise_FFT_Lists = [(2*abs(np.fft.fftshift(np.fft.fft(N_data)))**2/int(len(N_data)*sample_rate))\
                           [int(len(N_data)/2)+2:]
                           for N_data in Noise_list]
        Fre_Lists =[(np.linspace(0,sample_rate,len(N_data))-
                     int(len(N_data)/2)/((len(N_data)-1)/sample_rate))\
                    [int(len(N_data)/2)+2:]
                    for N_data in Noise_list]
    if is_equal_avg:
        """使用等间隔分割平均"""
        FN_all = [PSD_Avg_Equal(f,n,interval_num) for f,n in zip(*[Fre_Lists,Noise_FFT_Lists])]
        Fre_Avg_Lists, PSD_Avg_lists = list(map(list,zip(*FN_all)))
    else:
        FN_all = [FFT_Avg_Order(f,n,order_base,order_num) for f,n in zip(*[Fre_Lists,Noise_FFT_Lists])]
        Fre_Avg_Lists, PSD_Avg_lists = list(map(list,zip(*FN_all)))

    """ 多次噪声谱再平均"""
    Fre_Avg_cycle = list(map(np.mean,zip(*Fre_Avg_Lists)))
    PSD_Avg_cycle = list(map(np.mean,zip(*PSD_Avg_lists)))
    return({'Noise_FFT_Lists':Noise_FFT_Lists, 'Fre_Lists': Fre_Lists,
            'Fre_Avg_Interval':Fre_Avg_Lists, 'PSD_Avg_Interval': PSD_Avg_lists,
            'Fre_Avg_cycle': Fre_Avg_cycle, 'PSD_Avg_cycle': PSD_Avg_cycle})

""" 定义将V^2/Hz 数据转化为dBm/Hz  10*lg(Vn^2/4R/1mW)"""
def Svv_to_dBm(P_s, # 噪声谱功率 V^2/Hz
               R = 50,
              ):
    S_dBm = 10*np.log10(P_s/1e-3/(4*R))
    return(S_dBm)

"""npy数据提取函数"""
def Get_npy_datas(file_name):
    data_list = []
    with open(file_name, 'r', encoding = 'utf-8') as f_l:
        for row in csv.reader(f_l):
            data_list.append(row)
        # read_date = f_l.readline()
    return_data = list(map(lambda x: list(map(float,x)), data_list))
    return(return_data)

""" 定义csv数据提取函数 """
def Get_csv_data(file_name, S_index, E_index):
    data_list = []
    with open(file_name, 'r', encoding = 'utf-8') as f_l:
        for row in csv.reader(f_l):
            data_list.append(row)
        # read_date = f_l.readline()
    data_return = list(map(lambda x: list(map(float,x)), data_list[S_index:E_index]))
    return(data_return, #噪声谱数据
           data_list,   #csv所有数据
           )

""" 定义将频谱仪数据转化为等效温度函数 """
def Teff_func(P_s, # 噪声谱功率 dBm, P_s = V_n*I_n
              B_w,  # 采样带宽 Hz
              ):
    k_b= 1.3806505e-23
    kTB = 10**(P_s/10)*1e-3
    T_eff = kTB/(B_w*k_b)
    return(T_eff)

""" 定义将频谱仪数据转化为等效温度函数 """
def Svv_to_Teff(P_s, # 噪声谱功率 V^2/Hz
                 R = 50,
              ):
    P_dBm = Svv_to_dBm(P_s,R)
    T_eff = Teff_func(P_dBm,1)
    return(T_eff)

""" 定义将频谱仪数据转化为S_vv函数 """
def dBm_to_Svv(P_s, # 噪声谱功率 dBm, P_s = V_n*I_n
                 B_w,  # 采样带宽 Hz
              ):
    S_vv = (10**(P_s/10))*1e-3/B_w*4*50
    return(S_vv)

"""定义噪声谱拟合函数"""
def Fit_PSD_func(f,A_1,B,A_2,mu,sigma):
    return(np.log(A_1/f**1+A_2*sigma/((f-mu)**2-sigma**2)+B))

def Fit_PSD_1f_log_func(f,A,alpha,B):
    return(np.log(A/f**alpha++B))

"""定义smooth函数"""
def smooth(x,num):
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


"""定义数据汇总函数"""
def Noise_process_func(
        load_name_list,
        SA_start_point = 0):
    load_DMM,load_SR860,load_SA = load_name_list
    DMM_Data_all = np.load(load_DMM,allow_pickle=True).tolist() 
    SR_Data_all = np.load(load_SR860,allow_pickle=True).tolist()
    """万用表数据处理"""
    
    NPLC_list = [D_i['NPLC'] for D_i in DMM_Data_all]  #积分时间列表
    DMM_data_list = [D_i['Data_list'] for D_i in DMM_Data_all]

    Sample_rate_dict = {'5': 10, '1': 48, '0.5': 96,
                        '0.1': 490, '0.05': 960, '0.01': 4800}      #NPLC与采样率的对应关系字典
    Sample_rate_list = [Sample_rate_dict[str(nplc)] for nplc in NPLC_list]   #采样率列表
    
    Freq_list_DMM = []
    PSD_list_DMM = []
    for ii in range(len(DMM_data_list)):
        sample_rate = Sample_rate_list[ii]
        Noise_DMM_data_i = DMM_data_list[ii]
        FFT_ZPA_datas_i = Noise_FFT_Avg(Noise_DMM_data_i,sample_rate = sample_rate,
                                        is_equal_avg = True, is_2side_psd = False)   #计算平均噪声谱
        """"排除第一个点为DC的情况"""
        Fre_Avg_ZPA_i = FFT_ZPA_datas_i['Fre_Avg_cycle'][1:]
        PSD_Avg_ZPA_i = FFT_ZPA_datas_i['PSD_Avg_cycle'][1:]
        
        Freq_list_DMM.append(Fre_Avg_ZPA_i)
        PSD_list_DMM.append(PSD_Avg_ZPA_i)

    Freq_list_DMM = sum(Freq_list_DMM,[])
    PSD_list_DMM = sum(PSD_list_DMM,[])

    # 使用zip将Freq和PSD列表合并成元组列表，然后按照Freq的值排序
    sorted_data_DMM = sorted(zip(Freq_list_DMM, PSD_list_DMM), key=lambda x: x[0])
    # 解压排序后的数据并更新Freq和PSD
    Freq_list_DMM, PSD_list_DMM = map(list, zip(*sorted_data_DMM))


    """锁相数据处理"""
    
    ENBW_list = [D_i['ENBW'] for D_i in SR_Data_all]  #返回的的ENBW数据似乎不准确
    """通过avg(R^2)/ENBW/4计算噪声谱密度大小"""
    SR_Data_lists = [list(map(eval,D_i['Data_list'])) for D_i in SR_Data_all]
    R_data_lists = np.array([list(zip(*D_i))[2] for D_i in SR_Data_lists])
    R2_mean_list = [R_i**2 for R_i in R_data_lists]

    ENBW = 0.026 #HZ
    PSD_list_SR860 = [np.mean(R2_mean_list[i]/ENBW/4) for i in range(len(R2_mean_list))]
    Freq_list_SR860 = [D_i['FREQ'] for D_i in SR_Data_all]

    """频谱仪数据处理"""
    """获取数据"""
    Data_SA, CSV_All_SA = Get_csv_data(load_SA,45,-1)
    SA_Awg_ZPA = list(map(list,zip(*Data_SA)))
    BW_ZPA = float(CSV_All_SA[11][-1])     #获取BW带宽大小
    # SA_50Ohm = list(map(list,zip(*Get_csv_data(file_50Ohm,45,-1))))

    Freq_list_SA = SA_Awg_ZPA[0]
    PSD_list_SA = [dBm_to_Svv(p,BW_ZPA) for p in SA_Awg_ZPA[1]]  #从dBm转换为V^2/Hz
    
    """噪声数据合并"""
    Freq_r = Freq_list_SR860+Freq_list_SA[SA_start_point:]+Freq_list_DMM
    PSD_r = PSD_list_SR860+PSD_list_SA[SA_start_point:]+PSD_list_DMM


    # 使用zip将Freq和PSD列表合并成元组列表，然后按照Freq的值排序
    sorted_data = sorted(zip(Freq_r, PSD_r), key=lambda x: x[0])

    # 解压排序后的数据并更新Freq和PSD
    Freq_r, PSD_r = map(np.array, zip(*sorted_data))
    
    return(
            dict(Freq_list_DMM = Freq_list_DMM, PSD_list_DMM = PSD_list_DMM,
                Freq_list_SR860 = Freq_list_SR860, PSD_list_SR860 = PSD_list_SR860,
                Freq_list_SA = Freq_list_SA, PSD_list_SA = PSD_list_SA,
                Freq_r = Freq_r, PSD_r = PSD_r)
           )

"""去除峰值"""
def Delete_peak_func(
        PSD_list):
    PSD_change_list = []
    for i in range(len(PSD_list)):
        psd = PSD_list[i]
        if i == 0:
            psd_change = psd
        elif (np.log10(psd)-np.log10(PSD_change_list[-1]))>0.15:
            print(i)
            psd_change = PSD_change_list[-1]
        else:
            psd_change = psd
        PSD_change_list.append(psd_change)
    return(np.array(PSD_change_list))
#%% 锁相数据处理
#读取数据
Data_all = np.load(r"C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\比特退相位噪声测试\Z板子噪声室温测试\EZQ2\锁相ezq2_z100mv.npy",
                   allow_pickle=True).tolist()
ENBW_list = [D_i['ENBW'] for D_i in Data_all]  #返回的的ENBW数据似乎不准确

"""通过avg(R^2)/ENBW/4计算噪声谱密度大小"""
Data_lists = [list(map(eval,D_i['Data_list'])) for D_i in Data_all]
R_data_lists = np.array([list(zip(*D_i))[2] for D_i in Data_lists])
R2_mean_list = [R_i**2 for R_i in R_data_lists]

ENBW = 0.026 #HZ
PSD_list_SR860 = [np.mean(R2_mean_list[i]/ENBW/4) for i in range(len(R2_mean_list))]
Freq_list_SR860 = [D_i['FREQ'] for D_i in Data_all]

"""
同曲线双y轴绘图，左边显示S_vv (V^2/Hz)，右边显示 dBm/Hz 
"""
fig,axes=plt.subplots()
axes.plot(Freq_list_SR860,PSD_list_SR860,label = 'PSD of ZPA by SR860')
# axes.scatter(sum(Fre_ZPA_Lists,[]),sum(PSD_OnlyZPA_Lists,[]), s=5,
#              label = 'AWG Noise ')
# axes.scatter(Fre_Avg_All[-1], PSD_Self_All[-1],s=5,
#             label = 'PSD of Zpa by DMM6500',c = 'y')
axes.set_xscale('log')
axes.set_yscale('log')
axes.set_ylabel(r'$S_{vv}~ (V^2/Hz)$')
axes.set_xlabel('Freq (HZ)')
# axes.set_ylim(10**-17,10**-7)
twin_axes=axes.twinx() 
y1, y2 = axes.get_ylim() 
twin_axes.set_ylim(Svv_to_dBm(y1),Svv_to_dBm(y2))
# twin_axes.plot(Fre_sx,S_sx, label = 'SR860 datas')
twin_axes.set_ylabel('PSD (dBm/Hz)')
axes.legend()
# twin_axes.plot(SA_Awg_ZPA[0][10:],[p - 10*(np.log10(5.1e3)) for p in SA_Awg_ZPA[1]][10:], 
#          label = 'PSD of Zpa by \nSpectrum Analyzer ',c = 'r')
twin_axes.legend(loc='upper left')

plt.show()

#%% 频谱仪数据处理


folder_name = r'C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\比特退相位噪声测试\Z板子噪声室温测试\EZQ2_300mv'


file_ZPA = folder_name + r'\频谱仪EZQ2_Z300mV.csv'
BW = 5.1

"""获取数据"""
Data_ZPA, CSV_All_ZPA = Get_csv_data(file_ZPA,45,-1)
SA_Awg_ZPA = list(map(list,zip(*Data_ZPA)))
BW_ZPA = float(CSV_All_ZPA[11][-1])     #获取BW带宽大小
# SA_50Ohm = list(map(list,zip(*Get_csv_data(file_50Ohm,45,-1))))

Freq_list_SA = SA_Awg_ZPA[0]
PSD_list_SA = [dBm_to_Svv(p,BW_ZPA) for p in SA_Awg_ZPA[1]]  #从dBm转换为V^2/Hz

"""
存在前置放大时，AWG噪声与50本底噪声对比
同曲线双y轴绘图，左边显示噪声谱，右边显示S_vv
"""
fig,axes=plt.subplots()
twin_axes=axes.twinx() 
axes.plot(Freq_list_SA,PSD_list_SA, 
         label = 'PSD of ZPA by SA')

axes.set_xscale('log')
axes.set_yscale('log')
y1, y2 = axes.get_ylim() 
axes.set_ylabel('PSD (dBm/Hz)')
axes.set_xlabel('Freq (HZ)')
twin_axes.set_ylim(Svv_to_dBm(y1),Svv_to_dBm(y2))
twin_axes.set_ylabel(r'$S_{vv}~ (V^2/Hz)$')
# twin_axes.set_yscale('log')
axes.legend()
plt.show()
#%%万用表数据处理
Data_all = np.load(r"C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\比特退相位噪声测试\Z板子噪声室温测试\EZQ2\万用表ezq2_z100mv.npy",
                    allow_pickle=True).tolist()
# Data_all = np.load(r"D:\Documents\中科大\超导量子计算科研\实验\ZCZ3\ZCZ3_HF1\Chip2_KC0586-D1_KC9195-D4\Z线噪声谱测试\20240525/Unit5-08_-0.12.npy",
#                     allow_pickle=True).tolist()
# Data_all = np.load(r"D:\Documents\中科大\超导量子计算科研\实验\ZCZ3\ZCZ3_HF1\Chip2_KC0586-D1_KC9195-D4\Z线噪声谱测试\20240511/Unit4-10_-0.12.npy",
#                     allow_pickle=True).tolist()
NPLC_list = [D_i['NPLC'] for D_i in Data_all]  #积分时间列表

ZPA_data_list = [D_i['Data_list'] for D_i in Data_all]

Sample_rate_dict = {'5': 10, '1': 48, '0.5': 96,
                    '0.1': 490, '0.05': 960, '0.01': 4800}      #NPLC与采样率的对应关系字典

# Sample_rate_dict = {'10': 1/0.2, '1': 1/0.02, '0.2': 1/3e-3,
#                     '0.06': 1/1e-3, '0.02': 1/0.3e-3, '0.01': 4800}      #NPLC与采样率的对应关系字典
Sample_rate_list = [Sample_rate_dict[str(nplc)] for nplc in NPLC_list]   #采样率列表
"""提取数据并处理"""
# plt.figure()
Freq_list_DMM = []
PSD_list_DMM = []
Fre_50Ohm_Lists = []
PSD_50Ohm_Lists = []
for ii in range(len(ZPA_data_list)):
    sample_rate = Sample_rate_list[ii]
    Noise_ZPA_data_i = ZPA_data_list[ii]
    FFT_ZPA_datas_i = Noise_FFT_Avg(Noise_ZPA_data_i,sample_rate = sample_rate,
                                    is_equal_avg = True, is_2side_psd = False)   #计算平均噪声谱
    """"排除第一个点为DC的情况"""
    Fre_Avg_ZPA_i = FFT_ZPA_datas_i['Fre_Avg_cycle'][1:]
    PSD_Avg_ZPA_i = FFT_ZPA_datas_i['PSD_Avg_cycle'][1:]
    
    Freq_list_DMM.append(Fre_Avg_ZPA_i)
    PSD_list_DMM.append(PSD_Avg_ZPA_i)

Freq_list_DMM = sum(Freq_list_DMM,[])
PSD_list_DMM = sum(PSD_list_DMM,[])


# 使用zip将Freq和PSD列表合并成元组列表，然后按照Freq的值排序
sorted_data = sorted(zip(Freq_list_DMM, PSD_list_DMM), key=lambda x: x[0])

# 解压排序后的数据并更新Freq和PSD
Freq_list_DMM, PSD_list_DMM = map(list, zip(*sorted_data))

"""
同曲线双y轴绘图，左边显示噪声谱，右边显示S_vv
"""
fig,axes=plt.subplots()
axes.plot(Freq_list_DMM,[p for p in PSD_list_DMM], 
             label = 'PSD of ZPA by EZQ1 ')
axes.set_xscale('log')
axes.set_yscale('log')
axes.set_ylabel(r'$S_{vv}~ (V^2/Hz)$')
axes.set_xlabel('Freq (HZ)')
axes.legend()

# twin_axes=axes.twinx() 
# y1, y2 = axes.get_ylim() 
# twin_axes.set_ylim(Svv_to_dBm(y1),Svv_to_dBm(y2))
# twin_axes.set_ylabel('PSD (dBm/Hz)')
plt.show()

#%% 噪声合并
Freq_r = Freq_list_SR860+Freq_list_SA[30:]+Freq_list_DMM
PSD_r = PSD_list_SR860+PSD_list_SA[30:]+PSD_list_DMM
# 使用zip将Freq和PSD列表合并成元组列表，然后按照Freq的值排序
sorted_data = sorted(zip(Freq_r, PSD_r), key=lambda x: x[0])
# 解压排序后的数据并更新Freq和PSD
Freq_r, PSD_r = map(np.array, zip(*sorted_data))

"""去除峰值和平滑"""
PSD_r_1 = Delete_peak_func(PSD_r)
PSD_r_s = smooth(PSD_r_1,3)
func_inter=interpolate.UnivariateSpline(np.log(Freq_r),[np.log(psd) for psd in PSD_r_s],k=5,s=7)

PSD_smooth=np.exp(func_inter(np.log(Freq_r)))

"""绘图"""
fig,axes=plt.subplots()
twin_axes=axes.twinx() 
axes.plot(Freq_list_SR860,PSD_list_SR860,
              label = 'PSD of ZPA by SR860 ')
axes.plot(Freq_list_SA[30:],PSD_list_SA[30:],
              label = 'PSD of ZPA by SA')
axes.plot(Freq_list_DMM,PSD_list_DMM, 
              label = 'PSD of ZPA by DMM6500')

axes.plot(Freq_r,PSD_smooth, 
              label = 'PSD after smooth')
axes.set_xscale('log')
axes.set_yscale('log')
y1, y2 = axes.get_ylim() 
axes.set_ylabel(r'$S_{vv}~ (V^2/Hz)$')
axes.set_xlabel('Freq (HZ)')
twin_axes.set_ylim(Svv_to_dBm(y1),Svv_to_dBm(y2))
# twin_axes.plot(Fre_sx,S_sx, label = 'SR860 datas')
twin_axes.set_ylabel('PSD (dBm/Hz)')
# twin_axes.set_yscale('log')
axes.legend()
# twin_axes.legend(loc='upper left')
# plt.text(50,-100,'50.2,-102.4')
# plt.text(50,-100,'50.2,-102.4')
plt.show()

#%% 噪声谱比较
load_DMM_EZQ1_300mV = r"C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\比特退相位噪声测试\Z板子噪声室温测试\EZQ1\万用表ezq1z300mv.npy"
load_SR_EZQ1_300mV = r"C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\比特退相位噪声测试\Z板子噪声室温测试\EZQ1\锁相ezq1z300mv.npy"
load_SA_EZQ1_300mV =  r"C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\比特退相位噪声测试\Z板子噪声室温测试\EZQ1\频谱仪ezq1z300mv.csv"

load_DMM_EZQ1_200mV = r"C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\比特退相位噪声测试\Z板子噪声室温测试\EZQ1\万用表ezq1z200mv.npy"
load_SR_EZQ1_200mV = r"C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\比特退相位噪声测试\Z板子噪声室温测试\EZQ1\锁相ezq1z200mv.npy"
load_SA_EZQ1_200mV =  r"C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\比特退相位噪声测试\Z板子噪声室温测试\EZQ1\频谱仪ezq1z200mv.csv"

load_DMM_EZQ1_100mV = r"C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\比特退相位噪声测试\Z板子噪声室温测试\EZQ1\万用表ezq1z100mv.npy"
load_SR_EZQ1_100mV = r"C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\比特退相位噪声测试\Z板子噪声室温测试\EZQ1\锁相ezq1z100mv.npy"
load_SA_EZQ1_100mV =  r"C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\比特退相位噪声测试\Z板子噪声室温测试\EZQ1\频谱仪ezq1z100mv.csv"


EZQ1_300mV_lists = [load_DMM_EZQ1_300mV, load_SR_EZQ1_300mV, load_SA_EZQ1_300mV]
EZQ1_datas_300mV = Noise_process_func(EZQ1_300mV_lists,SA_start_point=40)

EZQ1_200mV_lists = [load_DMM_EZQ1_200mV, load_SR_EZQ1_200mV, load_SA_EZQ1_200mV]
EZQ1_datas_200mV = Noise_process_func(EZQ1_200mV_lists,SA_start_point=25)

EZQ1_100mV_lists = [load_DMM_EZQ1_100mV, load_SR_EZQ1_100mV, load_SA_EZQ1_100mV]
EZQ1_datas_100mV = Noise_process_func(EZQ1_100mV_lists,SA_start_point=15)

Freq_EZQ1_300mV = EZQ1_datas_300mV['Freq_r']
PSD_EZQ1_300mV = np.array(EZQ1_datas_300mV['PSD_r'])

Freq_EZQ1_200mV = EZQ1_datas_200mV['Freq_r']
PSD_EZQ1_200mV = np.array(EZQ1_datas_200mV['PSD_r'])

Freq_EZQ1_100mV = EZQ1_datas_100mV['Freq_r']
PSD_EZQ1_100mV = EZQ1_datas_100mV['PSD_r']

fig,axes=plt.subplots()
twin_axes=axes.twinx() 
axes.plot(Freq_EZQ1_300mV,PSD_EZQ1_300mV, label = 'PSD of EZQ1-300mV')
axes.plot(Freq_EZQ1_200mV,PSD_EZQ1_200mV, label = 'PSD of EZQ1-200mV')
axes.plot(Freq_EZQ1_100mV,PSD_EZQ1_100mV, label = 'PSD of EZQ1-100mV')
axes.set_xscale('log')
axes.set_yscale('log')
y1, y2 = axes.get_ylim() 
axes.set_ylabel(r'$S_{vv}~ (V^2/Hz)$')
axes.set_xlabel('Freq (HZ)')
twin_axes.set_ylim(Svv_to_dBm(y1),Svv_to_dBm(y2))
# twin_axes.plot(Fre_sx,S_sx, label = 'SR860 datas')
twin_axes.set_ylabel('PSD (dBm/Hz)')
# twin_axes.set_yscale('log')
axes.legend()
# twin_axes.legend(loc='upper left')
plt.show()

#%% 计算比特T2

"""对原始数据进行smooth平均后再利用插值形成噪声谱函数形式"""
# Freq_r = Freq_EZQ1_100mV
# PSD_r = PSD_EZQ1_100mV

"""去除峰值"""
PSD_r_1 = Delete_peak_func(PSD_r)
PSD_r_s = smooth(PSD_r_1,3)
func_inter=interpolate.UnivariateSpline(np.log(Freq_r),[np.log(psd) for psd in PSD_r_s],k=5,s=7)
PSD_inter=np.exp(func_inter(np.log(Freq_r)))

"""对低于1HZ的数据进行拟合得到1/f噪声的拟合结果"""
index_1f = np.where(np.array(Freq_r)<1)[0]
amp_0 = ((PSD_r_s[index_1f][0] - PSD_r_s[index_1f][-1])/
         (1/Freq_r[index_1f][0] - 1/Freq_r[index_1f][-1]))
fit_1f_data,fit_1f_error = optimize.curve_fit(Fit_PSD_1f_log_func,Freq_r[index_1f],
                                        [np.log(psd) for psd in PSD_r_s[index_1f]], 
                                        [amp_0,1,PSD_r_s[index_1f][-1]], 
                                        maxfev = 400000)
PSD_1f_fit = np.exp(Fit_PSD_1f_log_func(Freq_r,*fit_1f_data))

"""对高于4MHZ的数据进行平均得到白噪声结果"""
index_w = np.where(np.array(Freq_r)>10e6)[0]
PSD_w = np.mean(PSD_r[index_w])

"""
拟合数据和白噪声转折点，
寻找符合区间内拟合值(或平均得到的白噪值)与插值数据最靠近的点作为转折点，
可以避免曲线出现突然的变化
"""
transf_1f = np.argmin(np.abs(np.log10(PSD_inter[index_1f])-np.log10(PSD_1f_fit[index_1f])))
transf_w = np.argmin(np.abs(np.log10(PSD_inter)-np.log10(PSD_w)))


def PSD_func(f):
    """
    生成噪声谱函数，
    在低于1/f噪声转折点使用1/f拟合结果生成噪声谱数据，
    在高于白噪声转折点使用10MHz以上的平均数值生成噪声数据，
    在两个转折点之间则使用插值函数生成噪声数据。
    """
    if f < Freq_r[transf_1f]:
        psd = np.exp(Fit_PSD_1f_log_func(f,*fit_1f_data))
    elif f > Freq_r[transf_w]:
        psd = PSD_w
    else:
        psd = np.exp(func_inter(np.log(f)))
    return(abs(psd))

# def PSD_func(f):
#     """
#     生成噪声谱函数，
#     在低于1/f噪声转折点使用1/f拟合结果生成噪声谱数据，
#     在高于白噪声转折点使用4MHz以上的平均数值生成噪声数据，
#     在两个转折点之间则使用插值函数生成噪声数据。
#     """
#     # if f < Freq_r[transf_1f]:
#     psd = np.exp(Fit_PSD_1f_log_func(f,*np.array([5.96702005e-11, 1, 0])))
#     # elif f > Freq_r[transf_w]:
#     #     psd = PSD_w
#     # else:
#     #     psd = np.exp(func_inter(f))
#     return(psd)

Freq_inter = np.linspace(np.log(np.min(Freq_r)*0.01), np.log(np.max(Freq_r)*100), 4000)
Freq_inter = np.exp(Freq_inter)
func_inter2=interpolate.UnivariateSpline(Freq_inter,
                                        [np.log(PSD_func(freq)) for freq in Freq_inter],k=2,s=30)
def PSD_func2(f):
    psd = np.exp(func_inter2(f))
    return psd
# def PSD_func2(f):
#     psd = np.exp(Fit_PSD_1f_log_func(f,10e3,3.5,PSD_w))
#     return psd
fig,axes=plt.subplots()
twin_axes=axes.twinx() 
# axes.plot(Freq_r,PSD_r, label = 'AWG PSD')
axes.plot(Freq_r,PSD_r_s, label = 'AWG PSD')

axes.plot(Freq_inter,[PSD_func(freq) for freq in Freq_inter])
axes.set_xscale('log')
axes.set_yscale('log')
y1, y2 = axes.get_ylim() 
axes.set_ylabel(r'$S_{vv}~ (V^2/Hz)$')
axes.set_xlabel('Freq (HZ)')
twin_axes.set_ylim(Svv_to_dBm(y1),Svv_to_dBm(y2))
# twin_axes.plot(Fre_sx,S_sx, label = 'SR860 datas')
twin_axes.set_ylabel('PSD (dBm/Hz)')
# twin_axes.set_yscale('log')
axes.legend()
# twin_axes.legend(loc='upper left')
plt.show()

#%%
kv_c_16M = 38260057580.079124  #partial(f)/partial(V)  coupler在16M耦合处的频率随电压变化的斜率
kv_c_off = 20148689377.66139  #partial(f)/partial(V)  coupler在关断点处的频率随电压变化的斜率
leakage_C16M_ZCZ3 = 0.0216                #ZCZ3 比特在打开16M耦合处的在coupler上的态泄露
leakage_Coff_ZCZ3 = 0.00615                    #ZCZ3 比特在关断点的在coupler上的态泄露
kv_q_300M = 2214117455.179478

# kv_c_16M = 58997285148.46638/np.sqrt(10**(-21/10))  #partial(f)/partial(V)  coupler在20M耦合处的频率随电压变化的斜率
# kv_c_off = 29950204279.18776/np.sqrt(10**(-21/10))  #partial(f)/partial(V)  coupler在关断点处的频率随电压变化的斜率
# kv_q_300M = 3036969283.1584916/np.sqrt(10**(-33/10))

def PSD_omega_16M(omega):
    psd_16M = PSD_func2(omega/2/np.pi)*kv_c_16M**2*(2*np.pi)**2*leakage_C16M_ZCZ3
    psd_off = PSD_func2(omega/2/np.pi)*kv_c_off**2*(2*np.pi)**2*leakage_Coff_ZCZ3
    psd = psd_16M+1*psd_off
    return psd/2/np.pi
def PSD_omega_off(omega):
    # psd_16M = PSD_func(omega/2/np.pi)*kv_c_16M**2*(2*np.pi)**2*leakage_C16M_ZCZ3
    psd_off = PSD_func2(omega/2/np.pi)*kv_c_off**2*(2*np.pi)**2*leakage_Coff_ZCZ3
    psd = 4*psd_off
    return psd/2/np.pi
def PSD_omega_300M(omega):
    # psd_16M = PSD_func(omega/2/np.pi)*kv_c_16M**2*(2*np.pi)**2*leakage_C16M_ZCZ3
    psd_off = PSD_func(omega/2/np.pi)*kv_q_300M**2*(2*np.pi)**2
    psd = psd_off
    return psd/2/np.pi


N_list=[0]

t_list=np.linspace(0.02e-6,30e-6,201)
cal_Tphis=[]
Cal_T2_by_Sw_list=[]

plt.figure()
for i in range(len(N_list)):
    N=N_list[i]   
    starttime_i=int(time.time())
    
    cal_Tphi_n = CPMG_Tphi(
                  S_w = PSD_omega_300M,
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
    plt.plot(t_list/1e-6,cal_Tphi_n.fit_Tp_func(t_list,*fit_datas),
             label='N='+str(N)+rf': fited $T_2^{{1/e}} = {round(T2_e/1e-6,2)} \mu$s')
    endtime_i=int(time.time())
    print('N = '+str(N)+' running time: '+str(endtime_i-starttime_i)+' s.')
plt.ylabel(r'$e^{-\langle\chi_N(t)\rangle}$')
plt.xlabel(r'T /$\mu s$')
plt.legend()
plt.show()               

Omega_list=np.linspace(0.1,2e6,1000)*2*np.pi

fig,axes=plt.subplots()
twin_axes=axes.twinx() 
axes.plot(Freq_r,PSD_r, label = 'AWG PSD')


axes.plot(Freq_inter,[PSD_func(freq) for freq in Freq_inter])
axes.set_xscale('log')
axes.set_yscale('log')
y1, y2 = axes.get_ylim() 
axes.set_ylabel(r'$S_{vv}~ (V^2/Hz)$')
axes.set_xlabel('Freq (HZ)')
twin_axes.plot(Omega_list/2/np.pi,[cal_Tphis[0].filter_CPMG(o, 3e-6) for o in Omega_list],
          'r--', label=rf'Filter function $g_{{{N_list[0]}}}$')
twin_axes.plot(Omega_list/2/np.pi,[cal_Tphis[0].filter_CPMG(o, 5e-6) for o in Omega_list],
          'b--', label=rf'Filter function $g_{{{N_list[0]}}}$')
twin_axes.set_ylabel(r'$S_q^{th}(\omega)$   $Hz$')
twin_axes.set_xlabel(r'$\omega/2\pi$ MHz')

axes.legend()
twin_axes.legend(loc='upper left')
plt.show()

# fig, ax1 = plt.subplots()
# ax1.plot(Omega_list/1e6/2/np.pi,[cal_Tphis[0].filter_Sw_CPMG(o, 1.75e-6) for o in Omega_list],
#           'r--', label=rf'Filter function $g_{{{N_list[0]}}}$')
# ax1.plot(Omega_list/1e6/2/np.pi,[cal_Tphis[0].filter_Sw_CPMG(o, 1.85e-6) for o in Omega_list],
#           'b--', label=rf'Filter function $g_{{{N_list[0]}}}$')
# ax1.set_ylabel(r'$S_q^{th}(\omega)$   $Hz$')
# ax1.set_xlabel(r'$\omega/2\pi$ MHz')
# ax1.legend(loc='upper left')
#%% 4-8G噪声谱

load_folder = r"C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\Z板子噪声室温测试\4-8G"
load_EZQ2_woS_wPA = load_folder+r"\EZQ2_woSingnal_wPA.csv"
load_EZQ2_woS_woPA = load_folder+r"\EZQ2_woSingnal_woPA.csv"
load_EZQ1_woLo_wPA = load_folder+r"\EZQ1_woSingnal_wPA.csv"
load_EZQ1_woLo_woPA = load_folder+r"\EZQ1_woSingnal_woPA.csv"
load_EZQ1_5GLo_wPA = load_folder+r"\EZQ1_5GLocal_wPA.csv"
load_EZQ1_5GLo_woPA = load_folder+r"\EZQ1_5GLocal_woPA.csv"
load_50Ohm_wPA = load_folder+r"\50Ohm_wPA.csv"
load_50Ohm_woPA = load_folder+r"\50Ohm_woPA.csv"

load_list = [load_50Ohm_wPA, load_EZQ2_woS_wPA, load_EZQ1_woLo_wPA, load_EZQ1_5GLo_wPA]
# load_list = [load_50Ohm_woPA, load_EZQ2_woS_woPA, load_EZQ1_woLo_woPA, load_EZQ1_5GLo_woPA]
label_list = [r'5o Ohm load',
              r'EZQ2 without singnal',
              r'EZQ1 without LO',
              r'EZQ1 with 5G LO']
"""获取数据"""
Freq_list_all = []
PSD_list_all = []
for ii in range(len(load_list)):
    file_ii = load_list[ii]
    Data_ii, CSV_All_ii= Get_csv_data(file_ii,45,-1)
    SA_Awg_ii = list(map(list,zip(*Data_ii)))
    BW_ZPA_ii = float(CSV_All_ii[11][-1])     #获取BW带宽大小
    # SA_50Ohm = list(map(list,zip(*Get_csv_data(file_50Ohm,45,-1))))
    
    Freq_list_ii = SA_Awg_ii[0]
    PSD_list_ii = [dBm_to_Svv(p,BW_ZPA_ii) for p in SA_Awg_ii[1]]  #从dBm转换为V^2/Hz
    Freq_list_all.append(Freq_list_ii)
    PSD_list_all.append(PSD_list_ii)
    
"""
存在前置放大时，AWG噪声与50本底噪声对比
同曲线双y轴绘图，左边显示噪声温度，右边显示S_vv
"""
fig,axes=plt.subplots()
twin_axes=axes.twinx() 
for i in range(len(Freq_list_all)-1):
    ii = i+1
    axes.plot(Freq_list_all[ii],
              [psd - psd_base for psd,psd_base in 
               zip(PSD_list_all[ii],PSD_list_all[0])], 
             label = label_list[ii])

axes.set_xscale('log')
axes.set_yscale('log')
y1, y2 = axes.get_ylim() 
axes.set_ylabel('PSD (dBm/Hz)')
axes.set_xlabel('Freq (HZ)')
twin_axes.set_ylim(Svv_to_Teff(y1),Svv_to_Teff(y2))
twin_axes.set_ylabel(r'$T_{eff}~ (K)$')
twin_axes.set_yscale('log')
axes.legend()
plt.show()
