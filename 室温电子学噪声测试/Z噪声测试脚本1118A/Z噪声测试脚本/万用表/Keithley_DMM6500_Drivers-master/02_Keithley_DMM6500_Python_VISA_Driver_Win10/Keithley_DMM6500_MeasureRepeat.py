
#%% 连接DMMM6500万用表测量并保存数据
import numpy as np
import matplotlib.pyplot as plt
import pyvisa
import struct
import math
import time
import Keithley_DMM6500_VISA_Driver as kei
import csv


#===== MAIN PROGRAM STARTS HERE =====
rm = pyvisa.ResourceManager()	# Opens the resource manager and sets it to variable rm
DAQ_Inst_1 = "USB0::0x05E6::0x6500::04594930::0::INSTR"
# Instrument ID String examples...
#       LAN -> TCPIP0::134.63.71.209::inst0::INSTR
#       USB -> USB0::0x05E6::0x2450::01419962::INSTR
#       GPIB -> GPIB0::16::INSTR
#       Serial -> ASRL4::INSTR
timeout = 20000000
myFile = "dmm_functions.tsp"

DMM6500 = kei.DMM6500()
myID = DMM6500.Connect(rm, DAQ_Inst_1, timeout, 1, 1, 1)
t1 = time.time()

DMM6500.LoadScriptFile(myFile)
DMM6500.SendCmd("do_beep(1.0, 3500)")

DMM6500.SetMeasure_Function(DMM6500.MeasFunc.DCV)
DMM6500.SetMeasure_Range(1)
DMM6500.SetMeasure_NPLC(1)
DMM6500.SetMeasure_InputImpedance(DMM6500.InputZ.Z_10M)
DMM6500.SetMeasure_AutoZero(DMM6500.DmmState.OFF)
DMM6500.SetMeasure_FilterType(DMM6500.FilterType.REP)
DMM6500.SetMeasure_FilterCount(1)
DMM6500.SetMeasure_FilterState(DMM6500.DmmState.OFF)


Measure_list = []
Cycle_num = 3
M_num = 2e4
for c in range(Cycle_num):
    """清空DMM缓存内的之前测量数据"""
    DMM6500.Clear_Buffer()
    """重复测量 M_num 次数据"""
    DMM6500.Measure_repeat(M_num)
    """提取DMM缓存内的测量数据"""
    M_c = list(eval(DMM6500.Get_Buffer_Data()))
    
    Measure_list.append(M_c)
    
t2 = time.time()
print("done")
print("{0:.6f} s".format(t2-t1))
DMM6500.Disconnect()
rm.close()
#time.sleep(1.0)

"""保存测量数据"""
# Measure_list = [[1,2,3],[4,5,6]]
file = open(r'C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\Z板子噪声室温测试\DMM测试数据\Hefei_F111_6+\Measure_HeFei_50Ohm_NPLC1.csv', 
            mode='w', newline='')
writer = csv.writer(file)
for fp in Measure_list:
    writer.writerow(fp)
    # writer.writerow('\n')
file.close()

# t2 = time.time()

# Notify the user of completion and the test time achieved. 
# print("done")
# print("{0:.6f} s".format(t2-t1))
# input("Press Enter to continue...")
# exit()

# %% 测量数据处理 (FFT)
folder_name = r'C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\Z板子噪声室温测试\DMM测试数据'
file_ZpaN_5 =folder_name+r'\Measure_HeFei_F081_4_NPLC5.csv'
file_50Ohm_5 =folder_name+r'\Measure_HeFei_50Ohm_NPLC5.csv'
file_ZpaN_1 =folder_name+r'\Measure_HeFei_F081_4_NPLC1.csv'
file_50Ohm_1 =folder_name+r'\Measure_HeFei_50Ohm_NPLC1.csv'
file_ZpaN_05 =folder_name+r'\Measure_HeFei_F081_4_NPLC0-5.csv'
file_50Ohm_05 =folder_name+r'\Measure_HeFei_50Ohm_NPLC0-5.csv'
file_ZpaN_01 =folder_name+r'\Measure_HeFei_F081_4_NPLC0-1.csv'
file_50Ohm_01 =folder_name+r'\Measure_HeFei_50Ohm_NPLC0-1.csv'
"""数据提取函数"""
def Get_datas(file_name):
    data_list = []
    with open(file_name, 'r', encoding = 'utf-8') as f_l:
        for row in csv.reader(f_l):
            data_list.append(row)
        # read_date = f_l.readline()
    return_data = list(map(lambda x: list(map(float,x)), data_list))
    return(return_data)

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
        """双边噪声谱"""
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
    

File_ZPA_list = [file_ZpaN_5, file_ZpaN_1,file_ZpaN_05,file_ZpaN_01]
File_50Ohm_list = [file_50Ohm_5, file_50Ohm_1,file_50Ohm_05,file_50Ohm_01]
Sample_rate_list=[5,48,96,490]

Zpa_label = ['Zpa noise with NPLC = 1']

Fre_ZPA_Lists = []
PSD_ZPA_Lists = []
Fre_50Ohm_Lists = []
PSD_50Ohm_Lists = []
for ii in range(len(File_ZPA_list)):
    sample_rate = Sample_rate_list[ii]
    Noise_ZPA_data_i = Get_datas(File_ZPA_list[ii]) 
    Noise_50Ohm_data_i = Get_datas(File_50Ohm_list[ii]) 
    
    FFT_ZPA_datas_i = Noise_FFT_Avg(Noise_ZPA_data_i,sample_rate = sample_rate,
                                    is_equal_avg = True, is_2side_psd = False)
    FFT_50Ohm_datas_i = Noise_FFT_Avg(Noise_50Ohm_data_i,sample_rate = sample_rate,
                                      is_equal_avg = True, is_2side_psd = False)
    
    Fre_Avg_ZPA_i = FFT_ZPA_datas_i['Fre_Avg_cycle']
    PSD_Avg_ZPA_i = FFT_ZPA_datas_i['PSD_Avg_cycle']
    Fre_Avg_50Ohm_i = FFT_50Ohm_datas_i['Fre_Avg_cycle']
    PSD_Avg_50Ohm_i = FFT_50Ohm_datas_i['PSD_Avg_cycle']
    
    Fre_ZPA_Lists.append(Fre_Avg_ZPA_i)
    PSD_ZPA_Lists.append(PSD_Avg_ZPA_i)
    Fre_50Ohm_Lists.append(Fre_Avg_50Ohm_i)
    PSD_50Ohm_Lists.append(PSD_Avg_50Ohm_i)

"""计算扣去50Ohm本底噪声之后的ZPA 噪声谱"""
PSD_OnlyZPA_Lists = list(map(lambda x: [z-o for z,o in zip(*x)], zip(PSD_ZPA_Lists,PSD_50Ohm_Lists)))
    
# Noise_ZPA_data = Get_datas(file_ZpaN_5) 
# FFT_ZPA_datas = Noise_FFT_Avg(Noise_ZPA_data,smaple_rate = 5,is_2side_psd = False)
# Fre_Avg_ZPA = FFT_ZPA_datas['Fre_Avg_cycle']
# PSD_Avg_ZPA = FFT_ZPA_datas['PSD_Avg_cycle']
# Noise_50Ohm_data = Get_datas(file_50Ohm_5) 
# FFT_50Ohm_datas = Noise_FFT_Avg(Noise_50Ohm_data,smaple_rate = 5,is_2side_psd = False)
# Fre_Avg_50Ohm = FFT_50Ohm_datas['Fre_Avg_cycle']
# PSD_Avg_50Ohm = FFT_50Ohm_datas['PSD_Avg_cycle']



# plt.figure()
plt.scatter(sum(Fre_50Ohm_Lists,[]),sum(PSD_50Ohm_Lists,[]),s=5,
            label = 'PSD of 50Ohm load')
plt.scatter(sum(Fre_ZPA_Lists,[]),PSD_OnlyZPA_Lists, s=5,
            label = 'PSD of AWGF081-04')
# plt.scatter(sum(Fre_ZPA_Lists,[]),sum(PSD_ZPA_Lists,[]),s=5,
#             label = 'PSD of (AWGF081-04 + 50Ohm load)')

# plt.ylabel(r'PSD (dBm/Hz)')
plt.ylabel(r'$S_{vv} ~(V^2/Hz)$')
plt.xlabel(r'Freq (Hz)') 
plt.yscale('log')
plt.xscale('log')
plt.legend()


plt.figure()
for ii in range(len(Fre_50Ohm_Lists)):
    
    plt.scatter(Fre_50Ohm_Lists[ii],PSD_50Ohm_Lists[ii],s=5,
                label = 'PSD of 50Ohm load')
    plt.scatter(Fre_ZPA_Lists[ii],PSD_ZPA_Lists[ii], s=5,
                label = 'PSD of AWGF081-04')
# plt.scatter(sum(Fre_ZPA_Lists,[]),sum(PSD_ZPA_Lists,[]),s=5,
#             label = 'PSD of (AWGF081-04 + 50Ohm load)')

# plt.ylabel(r'PSD (dBm/Hz)')
plt.ylabel(r'$S_{vv} ~(V^2/Hz)$')
plt.xlabel(r'Freq (Hz)') 
plt.yscale('log')
plt.xscale('log')
plt.legend()

