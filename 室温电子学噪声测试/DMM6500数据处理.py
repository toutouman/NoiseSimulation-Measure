import numpy as np
import matplotlib.pyplot as plt
import math
import time
import csv
from scipy import interpolate, optimize

#%%处理函数定义
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

"""数据提取函数"""
def Get_datas(file_name):
    data_list = []
    with open(file_name, 'r', encoding = 'utf-8') as f_l:
        for row in csv.reader(f_l):
            data_list.append(row)
        # read_date = f_l.readline()
    return_data = list(map(lambda x: list(map(float,x)), data_list))
    return(return_data)

#%% 测试数据处理FFT
folder_name1 = r'C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\Z板子噪声室温测试\DMM测试数据\Hefei_F038_3'
folder_name2 = r'C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\Z板子噪声室温测试\DMM测试数据\Hefei_F111_6+'
"""测试数据文件"""
file_50Ohm_5 =folder_name1+r'\Measure_HeFei_50Ohm_NPLC5.csv'
file_50Ohm_1 =folder_name1+r'\Measure_HeFei_50Ohm_NPLC1.csv'
file_50Ohm_05 =folder_name1+r'\Measure_HeFei_50Ohm_NPLC0-5.csv'
file_50Ohm_01 =folder_name1+r'\Measure_HeFei_50Ohm_NPLC0-1.csv'
file_50Ohm_005 =folder_name1+r'\Measure_HeFei_50Ohm_NPLC0-05.csv'

# file_ZpaN_5 =folder_name2+r'\Measure_HeFei_F111_6+_0_NPLC5.csv'
# file_ZpaN_1 =folder_name2+r'\Measure_HeFei_F111_6+_0_NPLC1.csv'
# file_ZpaN_05 =folder_name2+r'\Measure_HeFei_F111_6+_0_NPLC0-5.csv'
# file_ZpaN_01 =folder_name2+r'\Measure_HeFei_F111_6+_0_NPLC0-1.csv'
# file_ZpaN_005 =folder_name2+r'\Measure_HeFei_F111_6+_0_NPLC0-05.csv'

file_ZpaN_5 =folder_name1+r'\Measure_HeFei_F038_3_5e3_NPLC5.csv'
file_ZpaN_1 =folder_name1+r'\Measure_HeFei_F038_3_5e3_NPLC1.csv'
file_ZpaN_05 =folder_name1+r'\Measure_HeFei_F038_3_5e3_NPLC0-5.csv'
file_ZpaN_01 =folder_name1+r'\Measure_HeFei_F038_3_5e3_NPLC0-1.csv'
file_ZpaN_005 =folder_name1+r'\Measure_HeFei_F038_3_5e3_NPLC0-05.csv'

Sample_rate_list = [10,48,96,490,960]
File_ZPA_list = [file_ZpaN_5, file_ZpaN_1,file_ZpaN_05,file_ZpaN_01,file_ZpaN_005]
File_50Ohm_list = [file_50Ohm_5, file_50Ohm_1,file_50Ohm_05,file_50Ohm_01,file_50Ohm_005]

"""提取数据并处理"""
# plt.figure()
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
    # plt.plot(FFT_ZPA_datas_i['Fre_Lists'][0],FFT_ZPA_datas_i['Noise_FFT_Lists'][0])
# plt.ylabel(r'$S_{vv} ~(V^2/Hz)$')
# plt.xlabel(r'Freq (Hz)') 
# plt.yscale('log')
# plt.xscale('log')
# plt.legend()



"""计算扣去50Ohm本底噪声之后的ZPA 噪声谱"""
PSD_OnlyZPA_Lists = list(map(lambda x: [z-o for z,o in zip(*x)], zip(PSD_ZPA_Lists,PSD_50Ohm_Lists)))
    



"""
同曲线双y轴绘图，左边显示噪声谱，右边显示S_vv
"""
fig,axes=plt.subplots()
twin_axes=axes.twinx() 
axes.scatter(sum(Fre_50Ohm_Lists,[]),sum(PSD_50Ohm_Lists,[]),s=5,
             label = 'PSD of 50Ohm load')
axes.scatter(sum(Fre_ZPA_Lists,[]),sum(PSD_OnlyZPA_Lists,[]), s=5,
             label = 'AWG Noise ')

# plt.plot(SA_Awg_MWS_off[0],SA_Awg_MWS_off[1], label = 'AWG PSD with LO off')
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
twin_axes.legend(loc='upper left')
plt.show()

"""对不同积分时间NPLC独一画图"""
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

# %% 测量数据处理 (多组FFT比较)
folder_name2 = r'C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\Z板子噪声室温测试\DMM测试数据\Hefei_F111_6+'
# folder_name1 = r'C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\Z板子噪声室温测试\DMM测试数据'
folder_name1 = r'C:\Users\mantoutou\OneDrive\文档\中科大\超导量子计算科研\实验\Z板子噪声室温测试\DMM测试数据\Hefei_F038_3'
# file_ZpaN_5 =folder_name1+r'\Measure_HeFei_F081_4_NPLC5.csv'
# file_ZpaN_1 =folder_name1+r'\Measure_HeFei_F081_4_NPLC1.csv'
# file_ZpaN_05 =folder_name1+r'\Measure_HeFei_F081_4_NPLC0-5.csv'
# file_ZpaN_01 =folder_name1+r'\Measure_HeFei_F081_4_NPLC0-1.csv'
# file_ZpaN_005 =folder_name1+r'\Measure_HeFei_F081_4_NPLC0-05.csv'

"""测试数据文件"""
file_50Ohm_5 =folder_name2+r'\Measure_HeFei_50Ohm_NPLC5.csv'
file_50Ohm_1 =folder_name2+r'\Measure_HeFei_50Ohm_NPLC1.csv'
file_50Ohm_05 =folder_name2+r'\Measure_HeFei_50Ohm_NPLC0-5.csv'
file_50Ohm_01 =folder_name2+r'\Measure_HeFei_50Ohm_NPLC0-1.csv'
file_50Ohm_005 =folder_name2+r'\Measure_HeFei_50Ohm_NPLC0-05.csv'

file_ZpaN0_5 =folder_name2+r'\Measure_HeFei_F111_6+_0_NPLC5.csv'
file_ZpaN0_1 =folder_name2+r'\Measure_HeFei_F111_6+_0_NPLC1.csv'
file_ZpaN0_05 =folder_name2+r'\Measure_HeFei_F111_6+_0_NPLC0-5.csv'
file_ZpaN0_01 =folder_name2+r'\Measure_HeFei_F111_6+_0_NPLC0-1.csv'
file_ZpaN0_005 =folder_name2+r'\Measure_HeFei_F111_6+_0_NPLC0-05.csv'

file_ZpaN1e4_5 =folder_name2+r'\Measure_HeFei_F111_6+_1e4_NPLC5.csv'
file_ZpaN1e4_1 =folder_name2+r'\Measure_HeFei_F111_6+_1e4_NPLC1.csv'
file_ZpaN1e4_05 =folder_name2+r'\Measure_HeFei_F111_6+_1e4_NPLC0-5.csv'
file_ZpaN1e4_01 =folder_name2+r'\Measure_HeFei_F111_6+_1e4_NPLC0-1.csv'
file_ZpaN1e4_005 =folder_name2+r'\Measure_HeFei_F111_6+_1e4_NPLC0-05.csv'

file_ZpaN2e4_5 =folder_name2+r'\Measure_HeFei_F111_6+_2e4_NPLC5.csv'
file_ZpaN2e4_1 =folder_name2+r'\Measure_HeFei_F111_6-_2e4_NPLC1.csv'
file_ZpaN2e4_05 =folder_name2+r'\Measure_HeFei_F111_6-_2e4_NPLC0-5.csv'
file_ZpaN2e4_01 =folder_name2+r'\Measure_HeFei_F111_6-_2e4_NPLC0-1.csv'
file_ZpaN2e4_005 =folder_name2+r'\Measure_HeFei_F111_6-_2e4_NPLC0-05.csv'



"""形成不同信号源的测试文件列表"""
File_ZPA0_list = [file_ZpaN0_5, file_ZpaN0_1,file_ZpaN0_05,file_ZpaN0_01,file_ZpaN0_005]
File_ZPA1e4_list = [file_ZpaN1e4_5, file_ZpaN1e4_1,file_ZpaN1e4_05,file_ZpaN1e4_01,file_ZpaN1e4_005]
File_ZPA2e4_list = [file_ZpaN2e4_5, file_ZpaN2e4_1,file_ZpaN2e4_05,file_ZpaN2e4_01,file_ZpaN2e4_005]
File_50Ohm_list = [file_50Ohm_5, file_50Ohm_1,file_50Ohm_05,file_50Ohm_01,file_50Ohm_005]
# File_ZPA5e3_list = [file_ZpaN5e3_5, file_ZpaN5e3_1,file_ZpaN5e3_05,file_ZpaN5e3_01,file_ZpaN5e3_005]
# Sample_rate_list=[10,48,96,490,980]

File_All = [File_50Ohm_list,File_ZPA0_list, File_ZPA1e4_list,File_ZPA2e4_list]
Samplt_rate_All = [[10,48,98,490,980]]*len(File_All)
Label_All = ['50 Ohm load', 'Noise with ZPA = 0',
             'Noise with ZPA = 1e4', 'Noise with ZPA = 2e4', 'Noise with ZPA = 5e3']

Fre_Avg_All = []
PSD_Avg_All = []
for f_l in range(len(File_All)):
    Sample_rate_l = Samplt_rate_All[f_l]
    Noise_data_list = [Get_datas(fl_ii) for fl_ii in File_All[f_l]]
    FFT_data_list = [Noise_FFT_Avg(Noise_data_list[ii],sample_rate = Sample_rate_l[ii],
                                    is_equal_avg = True, is_2side_psd = False) 
                     for ii in range(len(File_All[f_l]))]
    Fre_Avg_list = [FFT_data_list[ii]['Fre_Avg_cycle'] for ii in range(len(File_All[f_l]))]
    PSD_Avg_list = [FFT_data_list[ii]['PSD_Avg_cycle'] for ii in range(len(File_All[f_l]))]
    
    Fre_Avg_All.append(Fre_Avg_list)
    PSD_Avg_All.append(PSD_Avg_list)
    
"""计算扣去50Ohm本底噪声之后的ZPA 噪声谱"""
PSD_Self_Lists = [list(map(lambda x: [z-o for z,o in zip(*x)], zip(PSD_Avg_ii,PSD_Avg_All[0])))
                     for PSD_Avg_ii in PSD_Avg_All[1:]]
PSD_Self_All = [PSD_Avg_All[0], *PSD_Self_Lists]

"""
同曲线双y轴绘图，左边显示噪声谱，右边显示S_vv
"""

fig,axes=plt.subplots()
for ii in range(len(PSD_Self_All)):
    axes.scatter(Fre_Avg_All[ii], PSD_Self_All[ii],s=5,
                label = Label_All[ii])
axes.set_ylabel(r'$S_{vv} ~(V^2/Hz)$')
axes.set_xlabel(r'Freq (Hz)') 
axes.set_yscale('log')
axes.set_xscale('log')
axes.legend()
# twin_axes=axes.twinx() 
# y1, y2 = axes.get_ylim() 
# twin_axes.set_ylim(Svv_to_dBm(y1),Svv_to_dBm(y2))
# twin_axes.set_ylabel('PSD (dBm/Hz)')
# twin_axes.legend(loc='upper left')
plt.show()

#%% 定义噪声拟合函数
def Fit_Noise_func(f,alpha,A,B):
    return(np.log(A/f**alpha+B))
    
def Cal_PSD_FitNoise(Fre_list,   #输入的噪声频率序列 单位Hz
                     PSD_list,   #输入的电压噪声功率谱  单位V^2/HZ
                     Fit_Initial = None
                       ):
    need_indexs = np.where(np.array(PSD_list)>0)
    PSD_need = np.array(PSD_list)[need_indexs]
    Fre_need = np.array(Fre_list)[need_indexs]
    if not Fit_Initial:
        fmax_index = np.argmax(Fre_need)
        fmin_index = np.argmin(Fre_need)
        f_min = Fre_need[fmin_index]
        psd_fmin = PSD_need[fmin_index]
        
        # print(PSD_list)
        f_max = Fre_need[fmax_index]
        psd_fmax = PSD_need[fmax_index]
        alpha_ini = 1
        B_ini = psd_fmin
        A_ini = (psd_fmin - psd_fmax)/(1/f_min**alpha_ini-1/f_max**alpha_ini)
        Fit_Initial = [alpha_ini, A_ini, B_ini]
        # print(Fit_Initial)
    fit_n = optimize.curve_fit(Fit_Noise_func,Fre_need,np.log(PSD_need),p0 = Fit_Initial,maxfev = 80000)
    return(fit_n)
    
Fit_all = [Cal_PSD_FitNoise(sum(F_ii,[]),sum(P_ii,[])) for F_ii,P_ii in
                   zip(Fre_Avg_All, PSD_Self_All)]

fig,axes=plt.subplots()
for ii in range(len(PSD_Self_All)):
    Fre_fit_list = np.linspace(np.min(Fre_Avg_All[ii]),np.max(Fre_Avg_All[ii]),101)
    axes.scatter(Fre_Avg_All[ii], PSD_Self_All[ii],s=5,
                label = Label_All[ii])
    axes.plot(Fre_fit_list, [np.exp(Fit_Noise_func(f,*Fit_all[ii][0])) for f in Fre_fit_list],'k--')
axes.set_ylabel(r'$S_{vv} ~(V^2/Hz)$')
axes.set_xlabel(r'Freq (Hz)') 
axes.set_yscale('log')
axes.set_xscale('log')

twin_axes=axes.twinx() 
y1, y2 = axes.get_ylim() 
twin_axes.set_ylim(Svv_to_dBm(y1),Svv_to_dBm(y2))
axes.legend()
twin_axes.set_ylabel('PSD (dBm/Hz)')
# twin_axes.legend(loc='upper left')
plt.show()

