# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:00:32 2023

@author: mantoutou
"""
#%% 导入包
import numpy as np
import matplotlib.pyplot as plt
import pyvisa
import struct
import math
import time
import sys
# sys.path.append(r'D:\项目\Z噪声\万用表\Keithley_DMM6500_Drivers-master\02_Keithley_DMM6500_Python_VISA_Driver_Win10')
import Keithley_DMM6500_VISA_Driver as kei
import csv

# rm = pyvisa.ResourceManager()	# Opens the resource manager and sets it to variable rm
# rm.list_resources() # Get Instrument ID
# Instrument ID String examples...
#       LAN -> TCPIP0::134.63.71.209::inst0::INSTR
#       USB -> USB0::0x05E6::0x2450::01419962::INSTR
#       GPIB -> GPIB0::16::INSTR
#       Serial -> ASRL4::INSTR

#%% 连接DMM6500万用表测量并保存数据
#===== MAIN PROGRAM STARTS HERE =====
rm = pyvisa.ResourceManager()	# Opens the resource manager and sets it to variable rm
DAQ_Inst_1 = "USB0::0x05E6::0x6500::04594930::0::INSTR"

NPLC_list = [5,1,0.5,0.1,0.05]   #积分时间列表
Cycle_num = 3       #循环次数
M_num = 10e2         #每次循环采样点数
timeout = 50000*1000*2
myFile = "dmm_functions.tsp"
DMM6500 = kei.DMM6500()
myID = DMM6500.Connect(rm, DAQ_Inst_1, timeout, 1, 1, 1)
M_dict_lists = []
for ii in range(len(NPLC_list)):
    NPLC = NPLC_list[ii]

    t1 = time.time()
    
    DMM6500.LoadScriptFile(myFile)
    DMM6500.SendCmd("do_beep(1.0, 3500)")
    
    DMM6500.SetMeasure_Function(DMM6500.MeasFunc.DCV)
    DMM6500.SetMeasure_Range(1)        #设置测量量程  1V
    DMM6500.SetMeasure_NPLC(NPLC)      #设置积分时间NPLC  
    DMM6500.SetMeasure_InputImpedance(DMM6500.InputZ.Z_10M)   #内阻，只有10M欧姆选项
    DMM6500.SetMeasure_AutoZero(DMM6500.DmmState.OFF) 
    DMM6500.SetMeasure_FilterType(DMM6500.FilterType.REP)
    DMM6500.SetMeasure_FilterCount(1)
    DMM6500.SetMeasure_FilterState(DMM6500.DmmState.OFF)
    
    
    Measure_list = []
    
    for c in range(Cycle_num):
        """清空DMM缓存内的之前测量数据"""
        DMM6500.Clear_Buffer()
        """重复测量 M_num 次数据"""
        DMM6500.Measure_repeat(M_num)
        """提取DMM缓存内的测量数据"""
        M_c = list(eval(DMM6500.Get_Buffer_Data()))
        
        Measure_list.append(M_c)
        
    t2 = time.time()
    
    """返回数据"""
    dicts = {'Data_list': Measure_list, 'NPLC': NPLC,
             'Measure_num': M_num, 'Cycle_num': Cycle_num}
    M_dict_lists.append(dicts)
    print("done")
    print("{0:.6f} s".format(t2-t1))
DMM6500.Disconnect()
rm.close()
#time.sleep(1.0)

"""保存测量数据"""

np.save(r"D:\项目\Z噪声\万用表\data\万用表EOCZ19Z1-PD1V_L.npy", 
        M_dict_lists)
print("save .npy done")


#%% 数据处理
# dt = 1/48
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


Data_all = np.load(r"D:\项目\Z噪声\万用表\data\万用表EOCZ19Z1-PD1V_L.npy",
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


# %% data2csv
import csv
import pandas as pd
da = {'hz':Freq_list_DMM,'v2/hz':PSD_list_DMM}
df = pd.DataFrame(data=da)
df.to_csv(r"D:\项目\Z噪声\万用表\data\万用表EOCZ19Z1-PD1V_L.csv", index=False)



# %% 
import pyvisa
rm1=pyvisa.ResourceManager()
rm1.list_resources()


local_ins=rm1.open_resource('USB0::0x05E6::0x6500::04594930::INSTR')
print(local_ins.query('*IDN?'))

# %%
