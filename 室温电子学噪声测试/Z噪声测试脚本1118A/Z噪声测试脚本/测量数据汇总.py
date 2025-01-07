# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:16:38 2023

@author: mantoutou
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import csv
from scipy import interpolate, optimize

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

""" 定义将频谱仪数据转化为S_vv函数 """
def dBm_to_Svv(P_s, # 噪声谱功率 dBm, P_s = V_n*I_n
                 B_w,  # 采样带宽 Hz
              ):
    S_vv = (10**(P_s/10))*1e-3/B_w*4*50
    return(S_vv)

"""定义噪声谱拟合函数"""
def Fit_PSD_func(f,A_1,B,A_2,mu,sigma):
    return(np.log(A_1/f**1+A_2*sigma/((f-mu)**2-sigma**2)+B))


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
Data_all = np.load(r"D:\项目\Z噪声\频谱汇总\锁相EOCV2-Z9.npy",
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


folder_name = r'D:\项目\Z噪声\频谱汇总'


file_ZPA = folder_name + r'\频谱仪EOCV2-Z9.csv'

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

Data_all = np.load(r"D:\项目\Z噪声\频谱汇总\万用表EOCV2-Z9.npy",
                   allow_pickle=True).tolist()
NPLC_list = [D_i['NPLC'] for D_i in Data_all]  #积分时间列表

ZPA_data_list = [D_i['Data_list'] for D_i in Data_all]

Sample_rate_dict = {'5': 10, '1': 48, '0.5': 96,
                    '0.1': 490, '0.05': 960, '0.01': 4800}      #NPLC与采样率的对应关系字典
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
twin_axes=axes.twinx() 
axes.plot(Freq_list_DMM,PSD_list_DMM, 
             label = 'PSD of ZPA by DMM6500 ')
axes.set_xscale('log')
axes.set_yscale('log')
y1, y2 = axes.get_ylim() 
axes.set_ylabel(r'$S_{vv}~ (V^2/Hz)$')
axes.set_xlabel('Freq (HZ)')
twin_axes.set_ylim(Svv_to_dBm(y1),Svv_to_dBm(y2))
twin_axes.set_ylabel('PSD (dBm/Hz)')
axes.legend()
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
PSD_r_s = smooth(PSD_r_1,5)
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
plt.text(50,-100,'50.2,-102.4')
plt.text(50,-100,'50.2,-102.4')
plt.show()


# fig,axes=plt.subplots()
# twin_axes=axes.twinx() 
# axes.plot(Freq_list,PSD_list, label = 'AWG PSD')
# axes.plot(Freq_inter,PSD_inter)
# axes.plot(Freq_inter,[np.exp(Fit_PSD_func(freq,*fit_data)) for freq in Freq_inter])
# axes.set_xscale('log')
# axes.set_yscale('log')
# y1, y2 = axes.get_ylim() 
# axes.set_ylabel(r'$S_{vv}~ (V^2/Hz)$')
# axes.set_xlabel('Freq (HZ)')
# twin_axes.set_ylim(Svv_to_dBm(y1),Svv_to_dBm(y2))
# # twin_axes.plot(Fre_sx,S_sx, label = 'SR860 datas')
# twin_axes.set_ylabel('PSD (dBm/Hz)')
# # twin_axes.set_yscale('log')
# axes.legend()
# # twin_axes.legend(loc='upper left')
# plt.show()

# %%
import csv
import pandas as pd
da = {'hz':Freq_list,'v2/hz':PSD_list}
df = pd.DataFrame(data=da)
df.to_csv(r"D:\项目\Z噪声\频谱汇总\频谱汇总EOCV2-Z9.csv", index=False)



# %%
