# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:21:02 2024

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
#%% 连接万用表测量并保存数据

# 创建 Visa 对象
rm = pyvisa.ResourceManager()

# 根据实际情况更改VISA地址
visa_address =  'USB0::0x2A8D::0x0101::MY54506937::INSTR'

timeout = 50000*1000
try:
    # 打开设备
    dmm = rm.open_resource(visa_address, timeout=timeout)

    # 重置设备
    dmm.write('*RST')

    # 设置 NPLC 值（默认为 10）
    def set_nplc(nplc):
        dmm.write(f'VOLT:NPLC {nplc}')

    # 设置电压量程（默认为自动）
    def set_voltage_range(voltage_range):
        dmm.write(f'VOLT:RANG {voltage_range}')

    # 设置内阻（默认为自动）
    def set_impedance(impedance):
        dmm.write(f'VOLT:IMP:AUTO {impedance}')

    # 测量电压
    def repeat_measure_voltage(
            measure_num,
            voltage_range=1,
            nplc=1):
        # 执行测量
        # voltage = float(dmm.query(f'MEAS:VOLT:DC?{voltage_range},{1}'))
        dmm.write('CONF:VOLT:DC')
        dmm.write(f'VOLT:DC:RANG {voltage_range}')
        dmm.write(f'VOLT:DC:NPLC {nplc}')
        dmm.write(f'SAMP:COUN {measure_num}')
        voltage_str = dmm.query('READ?')
        voltage = list(eval(voltage_str))
    
        return voltage
    
        # 清除缓冲区
    def clear_buffer():
        dmm.write("*CLS")
        dmm.write("*RST")

    # 测量指定次数的电压值
    def measure_voltage_multiple_times(n):
        voltages = []
        for _ in range(n):
            voltage = measure_voltage()
            voltages.append(voltage)
            time.sleep(1)  # 等待一秒再进行下一次测量
        return voltages

    # 测试

    NPLC_list = [1,0.2,0.06,0.02]   #积分时间列表
    Cycle_num = 1       #循环次数
    M_num = 2e4         #每次循环采样点数
    M_dict_lists = []
    for ii in range(len(NPLC_list)):

        t1 = time.time()
        
        NPLC = NPLC_list[ii]
        # 设置 NPLC、电压量程和内阻
        set_voltage_range('1')  # 设置电压量程为1V
        set_impedance('10E6')  # 设置内阻为10M
        set_nplc(NPLC)  # 设置 NPLC 值为 10
        Measure_list = []
        for c in range(Cycle_num):
            """清空DMM缓存内的之前测量数据"""
            clear_buffer()
            """重复测量 M_num 次数据"""
            measured_voltages = repeat_measure_voltage(measure_num = int(M_num),
                                                       voltage_range = 1,
                                                       nplc = NPLC)
            Measure_list.append(measured_voltages)
            
        t2 = time.time()
        
        """返回数据"""
        dicts = {'Data_list': Measure_list, 'NPLC': NPLC,
                 'Measure_num': M_num, 'Cycle_num': Cycle_num}
        M_dict_lists.append(dicts)
        print(f"Mearment of NPLC = {NPLC} completed")
        print("{0:.6f} s".format(t2-t1))
        # print("Measured Voltages:", measured_voltages)
except pyvisa.VisaIOError as e:
    print("Error:", e)

    
finally:
    # 关闭设备
    dmm.close()
    rm.close()
"""保存测量数据"""

np.save(r"D:\Documents\中科大\超导量子计算科研\实验\ZCZ3\ZCZ3_HF1\Chip2_KC0586-D1_KC9195-D4\Z线噪声谱测试\20240525\Unit5-08_-0.12.npy", 
        M_dict_lists)
print("save .npy done")

#%%
import pyvisa
import time

# 创建 Visa 对象
rm = pyvisa.ResourceManager()

# 根据实际情况更改VISA地址
visa_address = 'USB0::0x2A8D::0x0101::MY54506937::INSTR'

try:
    # 打开设备
    dmm = rm.open_resource(visa_address)

    # 重置设备
    dmm.write('*RST')

    # 设置 NPLC 值（默认为 10）
    def set_nplc(nplc):
        dmm.write(f'VOLT:NPLC {nplc}')

    # 设置电压量程（默认为自动）
    def set_voltage_range(voltage_range):
        dmm.write(f'VOLT:RANG {voltage_range}')

    # 设置内阻（默认为自动）
    def set_impedance(impedance):
        dmm.write(f'VOLT:IMP:AUTO {impedance}')

    # 测量电压
    def measure_voltage():
        # 保存当前的 NPLC 和电压量程设置
        # saved_nplc = dmm.query('VOLT:NPLC?')
        # saved_range = dmm.query('VOLT:RANG?')
    
        # # 设置 NPLC 和电压量程为之前保存的值
        # dmm.write(f'VOLT:NPLC {saved_nplc}')
        # dmm.write(f'VOLT:RANG {saved_range}')
        voltage_range = 0.4
        nplc = 1
        # 执行测量
        # voltage = float(dmm.query(f'MEAS:VOLT:DC?{voltage_range},{1}'))
        dmm.write('CONF:VOLT:DC')
        dmm.write('VOLT:DC:RANG 1')
        dmm.write('VOLT:DC:NPLC 0.2')
        dmm.write('SAMP:COUN 5')
        voltage_str = dmm.query('READ?')
        # 恢复之前的 NPLC 和电压量程设置
        # dmm.write(f'VOLT:NPLC {saved_nplc}')
        # dmm.write(f'VOLT:RANG {saved_range}')
        voltage = list(eval(voltage_str))
        return voltage

    # 测量指定次数的电压值
    def measure_voltage_multiple_times(n):
        voltages = []
        for _ in range(n):
            # set_nplc(0.1)  # 设置 NPLC 值为 10
            # set_voltage_range(1)  # 设置电压量程为自动
            set_impedance('10E6')  # 设置内阻为自动
            voltage = measure_voltage()
            voltages.append(voltage)
            time.sleep(1)  # 等待一秒再进行下一次测量
        return voltages

    # 测试
    num_measurements = 2  # 指定次数
    measured_voltages = measure_voltage_multiple_times(num_measurements)
    print("Measured Voltages:", measured_voltages)

except pyvisa.VisaIOError as e:
    print("Error:", e)

# finally:
#     # 关闭设备
#     dmm.close()
#     rm.close()
