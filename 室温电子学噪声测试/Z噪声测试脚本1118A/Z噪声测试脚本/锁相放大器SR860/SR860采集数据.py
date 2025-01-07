# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 20:02:02 2023

@author: mantoutou
"""
#%%
import time
import pyvisa  # 用于与仪器通信的 VISA 库
import numpy as np
from threading import Event

# rm = pyvisa.ResourceManager()
# rm.list_resources() # Get Instrument ID

#%%数据采集
# 配置 VISA 接口，连接到 SR860 锁相放大器
rm = pyvisa.ResourceManager()
instrument_address = "USB0::0xB506::0x2000::003908::INSTR"
sr860 = rm.open_resource(instrument_address)

"""配置测量参数"""
measurement_time = 60*10   # 测量时间，单位是秒

"""配置 SR860 设置"""
sr860.timeout = 100*1000

sr860.write("OVRM 0")  # 使用内部参考源
sr860.write("OFLT 13")   #设置时间常数， 12:1s, 13:3s
sr860.write("OFSL 3")   #设置滤波器滚降，0:6dB, 1:12dB, 2:18dB, 3:24dB
sr860.write("HARM 1")  # 设置谐波模式为 1
sr860.write("SLVL 0.1")  # 设置参考信号电平，例如 0.1 V

sr860.write("CAPTURECFG 3")    #设置采集数据的形式， 3表示同时返回X,Y,R,theta信息
sr860.write("CAPTURELEN 256")  #设置采集数据的缓存大小， 128可以存储128*256/4组X,Y,Z,theta

#设置采集频率范围
M_freq_list = np.linspace(5**0.2,500e3**0.2,100)
M_freq_list = np.array(M_freq_list**5, dtype='int')

starttime=int(time.time())
M_dict_lists = []
for ii in range(len(M_freq_list)):
    sr860.write("IRNG 4")  # 设置输入信号的量程，0：1V， 1:300mV, 2:100mV, 3:30mV, 4:10mV 
    # if M_freq_list[ii]<10:
    #     sr860.write("IRNG 3")  # 设置输入信号的量程，0：1V， 1:300mV, 2:100mV, 3:30mV, 4:10mV 
    
    if M_freq_list[ii]<50:
        sr860.write("SCAL 16")  # 设置灵敏度范围，16 代表 5 uV rms 全量程
    # elif 10<=M_freq_list[ii]<50:
    #     sr860.write("SCAL 17")  # 设置灵敏度范围，17 代表 2 uV rms 全量程
    elif 50<=M_freq_list[ii]<500:
        sr860.write("SCAL 17")  # 设置灵敏度范围，18 代表 1 uV rms 全量程
    elif 500<=M_freq_list[ii]<2000:
        sr860.write("SCAL 18")  # 设置灵敏度范围，19 代表 0.5 uV rms 全量程
    elif 2000<=M_freq_list[ii]<50000:
        sr860.write("SCAL 19")  # 设置灵敏度范围，20 代表 0.2 uV rms 全量程
    else:
        sr860.write("SCAL 20")  # 设置灵敏度范围，21 代表 0.1 uV rms 全量程
    sr860.write("FREQ " + str(M_freq_list[ii]))  # 设置锁相放大器测量频率
    
    Max_rate= float(sr860.query("CAPTURERATEMAX?"))  #获取当前设置下的最大采样率 Hz
    n_rate = np.floor(np.log2(Max_rate/4)) if Max_rate>4 else Max_rate  # 采样率选择接近5HZ
    sr860.write("CAPTURERATE " + str(n_rate))   #设置采样率 Max_rate/2^n_rate
    
    # 启动测量
    print('开始测试频率为',str(M_freq_list[ii]),'Hz 信号 \n')
    sr860.write("REST")  # 重置锁相放大器
    sr860.write("CAPTURESTART 1,1")  # 启动数据采集，CAPTURESTART i,j . i=1表示连续采集， j=1表示使用硬件触发
    
    #delay时间等待测量完成，可以随时通过ctrl+C键在30s内停止
    time_delay = 0
    exit = Event()
    while not exit.is_set():
        exit.wait(30)
        time_delay = time_delay+30
        if time_delay >= measurement_time:
            exit.set()
    # 等待测量完成
    # time.sleep(measurement_time)  
    sr860.write("CAPTURESTOP")    #停止采集数据
    
    endtime=int(time.time())
    print('完成频率为',str(M_freq_list[ii]),'Hz信号测试，总耗时：', endtime-starttime, 's\n')
    
    M_rate= float(sr860.query("CAPTURERATE?"))
    """读取测量结果"""
    Data_list = []
    for i in range(2300):
        i_m = i + 20     #  跳过前面把别的频率被平均进去的数据
        Data_i = sr860.query("CAPTUREVAL? " + str(i_m))
        Data_list.append(Data_i)
    # X = float(sr860.query("OUTP? 0"))  # 读取通道 1 的测量结果，通常是 X 信号的幅值
    # Y = float(sr860.query("OUTP? 1"))  # 读取通道 2 的测量结果，通常是 Y 信号的幅值
    # R = float(sr860.query("OUTP? 2"))  # 读取通道 3 的测量结果，通常是 R 信号的幅值
    ENBW = float(sr860.query("ENBW?"))    #返回当前的滤波器的等效带宽(单边)
    SCAL = float(sr860.query("SCAL?"))
    IRNG = float(sr860.query("IRNG?"))
    dicts = {'FREQ': M_freq_list[ii], 'Data_list':Data_list, 'Measure_rate': M_rate, 
             'ENBW': ENBW, 'SCAL':SCAL, "IRNG": IRNG}
    M_dict_lists.append(dicts)

#保存测试结果
np.save(r"D:\项目\Z噪声\锁相放大器SR860\data\锁相EOCZ19Z1-BAT1V5.npy", M_dict_lists)
print("save .npy done")
# 关闭锁相放大器连接
sr860.close()


#%% 数据处理
import numpy as np
import matplotlib.pyplot as plt


def Svv_to_dBm(P_s, # 噪声谱功率 V^2/Hz
               R = 50,
              ):
    """ 
    定义将V^2/Hz 数据转化为dBm/Hz  即Pn = 10*lg(Vn^2/4R/1mW)
    """
    S_dBm = 10*np.log10(P_s/1e-3/(4*R))
    return(S_dBm)

#读取数据
Data_all = np.load(r"D:\项目\Z噪声\锁相放大器SR860\data\锁相EOCZ19Z1-BAT1V5.npy",
                   allow_pickle=True).tolist()
ENBW_list = [D_i['ENBW'] for D_i in Data_all]  #返回的的ENBW数据似乎不准确

"""通过avg(R^2)/ENBW/4计算噪声谱密度大小"""
Data_lists = [list(map(eval,D_i['Data_list'])) for D_i in Data_all]
R_data_lists = np.array([list(zip(*D_i))[2] for D_i in Data_lists])
R2_mean_list = [R_i**2 for R_i in R_data_lists]

ENBW = 0.026 #HZ
PSD_list = [np.mean(R2_mean_list[i]/ENBW/4) for i in range(len(R2_mean_list))]
Freq_list = [D_i['FREQ'] for D_i in Data_all]

"""
同曲线双y轴绘图，左边显示S_vv (V^2/Hz)，右边显示 dBm/Hz 
"""
fig,axes=plt.subplots()
axes.plot(Freq_list,PSD_list,label = 'PSD of 50Ohm load')
# axes.scatter(sum(Fre_ZPA_Lists,[]),sum(PSD_OnlyZPA_Lists,[]), s=5,
#              label = 'AWG Noise ')
axes.set_xscale('log')
axes.set_yscale('log')
axes.set_ylabel(r'$S_{vv}~ (V^2/Hz)$')
axes.set_xlabel('Freq (HZ)')

twin_axes=axes.twinx() 
y1, y2 = axes.get_ylim() 
twin_axes.set_ylim(Svv_to_dBm(y1),Svv_to_dBm(y2))
# twin_axes.plot(Fre_sx,S_sx, label = 'SR860 datas')
twin_axes.set_ylabel('PSD (dBm/Hz)')
axes.legend()
twin_axes.legend(loc='upper left')
plt.show()




# %%
import csv
import pandas as pd
da = {'hz':Freq_list,'v2/hz':PSD_list}
df = pd.DataFrame(data=da)
df.to_csv(r"D:\项目\Z噪声\锁相放大器SR860\data\锁相EOCZ19Z1-BAT1V5.csv", index=False)

# %%


