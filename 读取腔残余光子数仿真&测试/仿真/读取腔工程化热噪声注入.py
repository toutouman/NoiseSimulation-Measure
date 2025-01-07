# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:43:45 2023

@author: 馒头你个史剃磅
"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import interpolate
import time
import math
# from joblib import Parallel, delayed
#%% 参数定义

AWG_num = int(0.5e4)
AWG_rate = 2e9   # AWG 采样率
LO_rate = 10e9    # 微波源采样率
LO_Freq = 6e9   # LO 混频频率 Hz
phi = 0

Noise_power = 1     #噪声功率 dBm
LO_power = 20    #LO 功率 dBm
Noise_I = 0.5*np.random.randn(AWG_num)  #生成I通道白噪声 (标准差为1的高斯白噪声)
Noise_Q = np.random.randn(AWG_num)  #生产Q信号白噪声

T_awg_list = np.linspace(0,AWG_num/AWG_rate,AWG_num)  #AWG产生的时序
T_LO_list = np.arange(T_awg_list[0],T_awg_list[-1],1/LO_rate)  #LO产生的时序
F_awg_list = np.linspace(-AWG_rate/2,AWG_rate/2,AWG_num)    # AWG产生的频域

LO_num = len(T_LO_list)
LO_list_I = [LO_power*np.cos(LO_Freq*2*np.pi*n/LO_rate+phi) for n in range(LO_num)]  #生成I通道LO信号
LO_list_Q = [LO_power*np.sin(LO_Freq*2*np.pi*n/LO_rate+phi) for n in range(LO_num)]  #生成Q通道LO信号


"""绘制在AWG采样率下AWG信号噪声谱"""
plt.figure()
plt.subplot(211)
plt.hist(Noise_I,bins=50)
plt.subplot(212)
plt.psd(Noise_I,Fs = AWG_rate,
        sides = 'twosided',label ='PSD of AWG I chanel noise')
plt.legend()
#%%  混频噪声谱
start_time =time.time()

"""利用两种插值方法（线性式和阶梯式）将AWG信号转化为LO采样率下的信号"""        
kind_list = ['linear','zero']
NoiseI_Inter_Lists = []
NoiseQ_Inter_Lists = []

# plt.figure()
for kind in kind_list:
    """利用插值将AWG信号转化为LO采样率下的信号"""
    NoisI_inter_func = interpolate.interp1d(T_awg_list,Noise_I,kind=kind)
    NoisQ_inter_func = interpolate.interp1d(T_awg_list,Noise_Q,kind=kind)
    NoiseI_inter=NoisI_inter_func(T_LO_list)
    NoiseQ_inter=NoisQ_inter_func(T_LO_list)
    
    NoiseI_Inter_Lists.append(NoiseI_inter)
    NoiseQ_Inter_Lists.append(NoiseQ_inter)
    # plt.scatter(T_LO_list,NoiseI_inter,label=kind,s = 3)
# plt.legend()

NFFT = 512*2  # 将噪声谱频域分为若干个BLOCK，每个BLOCK内的PSD进行平均得到一个频点的PSD强度
"""绘制插值后的AWG信号噪声谱"""
plt.figure()
# pl.subplot(211)
# plt.hist(Noise_I,bins=50)
# plt.subplot(212)
plt.psd(NoiseI_Inter_Lists[0],Fs = LO_rate,
        sides = 'twosided',
        noverlap = 0,
        NFFT = NFFT,
        window= mlab.window_none,
        label ='PSD of wo MIX by ' + kind_list[0] + ' inter')
plt.psd(NoiseI_Inter_Lists[1],Fs = LO_rate,
        sides = 'twosided',
        noverlap = 0,
        NFFT = NFFT,
        window= mlab.window_none,
        label ='PSD of wo MIX by ' + kind_list[1] + ' inter')
plt.legend()



"""将插值后的IQ噪声信号与LO信号混频得到最终输出信号"""
Noist_all = [[NoiseI_Inter_Lists[k][n]*LO_list_I[n]+
              NoiseQ_Inter_Lists[k][n]*LO_list_Q[n] for n in range(LO_num)]
             for k in range(len(kind_list))]

"""绘制混频后的信号噪声谱"""
plt.figure()
plt.psd(Noist_all[0],Fs = LO_rate,
        sides = 'onesided',
        noverlap = 0,
        NFFT = NFFT,
        window= mlab.window_none,
        label ='PSD of with MIX 6GHz by ' + kind_list[0] + ' inter')
plt.psd(Noist_all[1],Fs = LO_rate,
        sides = 'onesided',
        noverlap = 0,
        NFFT = NFFT,
        window= mlab.window_none,
        label ='PSD of with MIX 6GHz by ' + kind_list[1] + ' inter')
plt.legend()

# """绘制混频后的信号时域"""
# plt.figure()
# plt.plot(Noist_all[0])
end_time =time.time()

print(rf'Cell run times: {np.round(end_time-start_time,1)} s')
#%% 采用不重叠窗口进行FFT
"""定义等间隔平均FFT 函数"""
def FFT_Avg_Equal(F_list,
                  PSD_list,
                  Interval_num = 200, #间隔数目
                  ):
    F_num = len(F_list)
    PSD_avg_list = []
    F_avg_list = []
    for ii in range(Interval_num):
        interval = int((F_num)/Interval_num)
        order_index = np.arange(ii*interval,(ii+1)*interval, dtype = 'int')
        # print(order_index)
        """返回PSD 单位为 dB/Hz"""
        PSD_avg_list.append(10*np.log10(np.mean(PSD_list[order_index])))
        F_avg_list.append(np.mean(F_list[order_index]))
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
            """返回PSD 单位为 dB/Hz"""
            F_avg.append(np.mean(F_list[indexs]))
            PSD_avg.append(10*np.log10(np.mean(PSD_list[indexs])))
    return(F_avg, PSD_avg)

NoiseI_FFT = abs(np.fft.fftshift(np.fft.fft(Noise_I)))**2/(AWG_rate*AWG_num)
NoiseQ_FFT = np.fft.fftshift(np.fft.fft(Noise_I))
"""将噪声谱按固定数量级平均化"""
O_max = np.ceil(np.log2(AWG_rate))
O_min = np.floor(np.log2(AWG_rate/(AWG_num-1))) 
Order_list = np.arange(O_min,O_max,1)

"""使用指数分割平均"""
F_awg_avg, NoiseI_FFT_avg = FFT_Avg_Order(F_awg_list,NoiseI_FFT,
                                          order_base=2, order_num = 10)

plt.figure()
plt.plot(F_awg_avg,NoiseI_FFT_avg)
plt.xscale('log')

"""使用等间隔分割平均"""
NoiseI_FFT_avg = []
F_awg_avg = []
F_awg_avg, NoiseI_FFT_avg = FFT_Avg_Equal(F_awg_list,NoiseI_FFT,Interval_num=200)

# Avg_num = 200
# for ii in range(Avg_num):
#     interval = int((AWG_num)/Avg_num)

#     order_index = np.arange(ii*interval,(ii+1)*interval,1)
#     NoiseI_FFT_avg.append(10*np.log10(np.mean(NoiseI_FFT[order_index])))
#     F_awg_avg.append(np.mean(F_awg_list[order_index]))
plt.figure()
plt.plot(F_awg_avg,NoiseI_FFT_avg)
# plt.xscale('log')

