# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:34:17 2023

@author: mantoutou
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(r'C:\Users\mantoutou\OneDrive\文档\程序\科大\约瑟夫森结哈密顿')
from package_define.Quibit_Cal_Function import *

def cal_kv_by_zpa2f01(f01, f_max, k_f, f_ah=-240e6):
    """从ZPA_mappper中的k_f计算比特频率相对电压的斜率"""
    zpa = np.arccos(((f01-f_ah)/(f_max-f_ah))**2)/(k_f*np.pi)
    print(zpa)
    k_z = 0.5*(f_max-f_ah)*(-np.pi*k_f*np.sin(np.pi*k_f*zpa)) / \
        np.sqrt(np.cos(np.pi*k_f*zpa))
    k_v = k_z*3.2e4/0.32
    return(k_v)


def cal_dB_by_target(Tphi, Tphi_t):
    decay_dB = 10*np.log10(Tphi_t/Tphi)
    return(decay_dB)


""" 定义将V^2/Hz 数据转化为dBm/Hz  10*lg(Vn^2/4R/1mW)"""
def Svv_to_dBm(P_s, # 噪声谱功率 V^2/Hz
               R = 50,
              ):
    S_dBm = 10*np.log10(P_s/1e-3/(4*R))
    return(S_dBm)

# %% couplerTphi1推算
"""计算P18_V3 coupler Q10 在5.4GHz 处的电压对频率的斜率 k_v"""
fmaxQ10_V3 = 9.03e9
fQ10_V3 = 5.4e9
fahQ10_V3 = -0.12e9
# WNoiseZPA_Q10_V3 = 1.9e6  # MHz  omega^2/Hz
WNoiseZPA_Q10_V3 = 1.7e6  # MHz  omega^2/Hz
zpa_mapperQ10_V3 = [7.35e9, -4e5, 1.849e-5]
ZMc_Q10_V3 = 1.07  # P18_V3 Coupler互感设计值 pH
# partial(f)/partial(V) Hz/V
kvQ10_V3 = cal_kv_by_zpa2f01(
    fQ10_V3, zpa_mapperQ10_V3[0], zpa_mapperQ10_V3[-1], fahQ10_V3)
phiQ10_V3, kphiQ10_V3 = flux_k_max_sim(
    fmaxQ10_V3, fahQ10_V3, fQ10_V3)  # 计算理论下比特的斜率k_phi (HZ/phi_0)
k_IVQ10_V3 = kvQ10_V3/kphiQ10_V3/ZMc_Q10_V3  # 计算Z线磁通电流I_z对板子电压V的斜率 I/V

WNoiseZPA_V_Q10_V3 = WNoiseZPA_Q10_V3 / \
    (2*np.pi)**2/kvQ10_V3**2  # 计算对应的板子噪声 V^2/Hz
# kphiQ10_V3 = kphiQ10_V3


"""计算ZCZ3 coupler 在打开20M 耦合处的T2大小"""
fmaxC_ZCZ3 = 10.5e9
f_C20M_ZCZ3 = 6.885e9  # ZCZ3.1 打开20M耦合频率
f_Coff_ZCZ3 = 9.31e9  # ZCZ3.1 Coupler关断点频率
fahC_ZCZ3 = -0.138e9
ZMc_C_ZCZ3 = 2.5  # ZCZ3.1 Coupler互感设计值 pH
# 计算ZCZ3 coupler 在打开20M 耦合处的斜率k_phi (HZ/phi_0)
phi_C20M_ZCZ3, kphi_C20M_ZCZ3 = flux_k_max_sim(
    fmaxC_ZCZ3, fahC_ZCZ3, f_C20M_ZCZ3)
phi_Coff_ZCZ3, kphi_Coff_ZCZ3 = flux_k_max_sim(
    fmaxC_ZCZ3, fahC_ZCZ3, f_Coff_ZCZ3)  # 计算ZCZ3 coupler 在关断点的斜率k_phi (HZ/phi_0)
# 推算ZCZ3 coupler 在打开20M 耦合处的白噪声  omega^2/Hz
WNoiseZPA_C20M_ZCZ3 = WNoiseZPA_V_Q10_V3 * \
    (k_IVQ10_V3*ZMc_C_ZCZ3)**2*kphi_C20M_ZCZ3**2*(2*np.pi)**2
WNoiseZPA_Coff_ZCZ3 = WNoiseZPA_V_Q10_V3 * \
    (k_IVQ10_V3*ZMc_C_ZCZ3)**2*kphi_Coff_ZCZ3**2 * \
    (2*np.pi)**2  # 推算ZCZ3 coupler 关断点的白噪声  omega^2/Hz
Tphi1_C20M_ZCZ3 = 4/WNoiseZPA_C20M_ZCZ3  # 推算ZCZ3 coupler 在打开20M 耦合处的T2
Tphi1_Coff_ZCZ3 = 4/WNoiseZPA_Coff_ZCZ3  # 推算ZCZ3 coupler 在关断点的T2


# leakage_C20M_ZCZ3 = 0.0216                #ZCZ3 比特在打开20M耦合处的在coupler上的态泄露
# leakage_Coff_ZCZ3 = 0.00615                    #ZCZ3 比特在关断点的在coupler上的态泄露
partialFreq_C20M_ZCZ3 = np.sqrt(0.0004382)  # ZCZ3 比特在打开20M耦合处频率随coupler变化的斜率
partialFreq_Coff_ZCZ3 = np.sqrt(1.09e-5)  # ZCZ3 比特在打开关断点处频率随coupler变化的斜率

Tphi1_Q20Ms_ZCZ3 = Tphi1_C20M_ZCZ3 / \
    partialFreq_C20M_ZCZ3**2  # 推算单个coupler在打开20M耦合时引起的比特T2限制
Tphi1_Qoffs_ZCZ3 = Tphi1_Coff_ZCZ3 / \
    partialFreq_Coff_ZCZ3**2  # 推算单个coupler在关断时引起的比特T2限制
Tphi1_Q20M_ZCZ3 = 4/(WNoiseZPA_C20M_ZCZ3*partialFreq_C20M_ZCZ3**2
                     + 3*WNoiseZPA_Coff_ZCZ3*partialFreq_Coff_ZCZ3**2)  # 一个打开耦合20M，三个关断引起的比特T2限制
Tphi1_Qoff_ZCZ3 = 4/(4*WNoiseZPA_Coff_ZCZ3 *
                     partialFreq_Coff_ZCZ3**2)  # 四个关断引起的比特T2限制
#%% coupler Tphi2推算

#EZQ1.1在低温21dB， 1.8e4码值的ZPA偏置下的P18_V3 coupler Tphi2 噪声 
# 不包含片上噪声和DC源0偏置噪声
FNoiseZPA_Q10_V3 = 5.64e6 
Tphi2_Q10_V3 = 1/FNoiseZPA_Q10_V3

Tphi2_C20M_ZCZ3 = Tphi2_Q10_V3*(ZMc_Q10_V3*kphiQ10_V3/ZMc_C_ZCZ3/kphi_C20M_ZCZ3)
Tphi2_Coff_ZCZ3 = Tphi2_Q10_V3*(ZMc_Q10_V3*kphiQ10_V3/ZMc_C_ZCZ3/kphi_Coff_ZCZ3)

Tphi2_Q20Ms_ZCZ3 = Tphi2_C20M_ZCZ3 / partialFreq_C20M_ZCZ3  # 推算单个coupler在打开20M耦合时引起的比特T2限制
Tphi2_Qoffs_ZCZ3 = Tphi2_Coff_ZCZ3 / partialFreq_Coff_ZCZ3**1  # 推算单个coupler在关断时引起的比特T2限制
Tphi2_Q20M_ZCZ3 = 1/np.sqrt(1/Tphi2_Q20Ms_ZCZ3**2
                     + 3*1/Tphi2_Qoffs_ZCZ3**2)  # 一个打开耦合20M，三个关断引起的比特T2限制
Tphi2_Qoff_ZCZ3 = 1/np.sqrt(4*1/Tphi2_Qoffs_ZCZ3**2)  # 四个关断引起的比特T2限制
# %% 比特Tphi1推算

""" 计算P18_V3 比特 Q01 在3.5GHz 处的电压对频率的斜率 k_v"""
fmaxQ01_V3 = 4.4354e9
fQ01_V3 = 3.511e9
fahQ01_V3 = -0.25e9
WNoiseZPA_Q01_V3 = 0.048e6  #omega^2/Hz
zpa_mapperQ01_V3 = [4.436e9, -0.91e4, 4.8e-6]
ZMc_Q01_V3 = 1.2  # P18_V3 Coupler互感设计值 pH
# ZMc_Q01_V3 = 1.2  # P18_V3 Coupler互感设计值 pH

# partial(f)/partial(V) Hz/V
kvQ01_V3 = cal_kv_by_zpa2f01(
    fQ01_V3, zpa_mapperQ01_V3[0], zpa_mapperQ01_V3[-1], fahQ01_V3)
phiQ01_V3, kphiQ01_V3 = flux_k_max_sim(
    fmaxQ01_V3, fahQ01_V3, fQ01_V3)  # 计算理论下比特的斜率k_phi (HZ/phi_0)
k_IVQ01_V3 = kvQ01_V3/kphiQ01_V3/ZMc_Q01_V3  # 计算Z线磁通电流I_z对板子电压V的斜率

WNoiseZPA_V_Q01_V3 = WNoiseZPA_Q01_V3 /(2*np.pi)**2/kvQ01_V3**2  # 计算对应的板子噪声 V^2/Hz


# fmaxQ01_V4 = 4.854e9
# fQ01_V4 = 4.079e9
# fahQ01_V4 = -0.25e9
# WNoiseZPA_Q01_V4 = 0.0285e6 #MHz
# zpa_mapperQ01_V4 = [4.833e9, -6.582e4, 3.674e-6]
# ZMc_Q01_V4 = 1.27  #P18_V3 Coupler互感设计值 pH
# kvQ01_V4 = cal_kv_by_zpa2f01(fQ01_V3, zpa_mapperQ01_V3[0], zpa_mapperQ01_V3[-1], fahQ01_V3)   #partial(f)/partial(V) Hz/V
# phiQ01_V4, kphiQ01_V3 = flux_k_max(fmaxQ01_V3, fahQ01_V3, fQ01_V3)  #计算理论下比特的斜率k_phi (HZ/phi_0)
# k_IVQ01_V4 = kvQ01_V3/kphiQ01_V3/ZMc_Q01_V3                         #计算Z线磁通电流I_z对板子电压V的斜率

"""计算ZCZ3 coupler 在偏置300M处的T2大小"""
fmaxQ_ZCZ3 = 5.277e9
fahQ_ZCZ3 = -0.239e9
ZMc_Q_ZCZ3 = 1.85  # ZCZ3.1 qubit互感设计值 pH
fidel_ZCZ3 = fmaxQ_ZCZ3-300E6  # ZCZ3.1 idel点频率
phi_Qidel_ZCZ3, kphi_Qidel_ZCZ3 = flux_k_max_sim(
    fmaxQ_ZCZ3, fahQ_ZCZ3, fidel_ZCZ3)  # 计算ZCZ3 比特在idel点的斜率k_phi (HZ/phi_0)
WNoiseZPA_Qidel_ZCZ3 = WNoiseZPA_V_Q01_V3 * \
    (k_IVQ01_V3*ZMc_Q_ZCZ3)**2*kphi_Qidel_ZCZ3**2 * \
    (2*np.pi)**2  # 推算ZCZ3 比特在idel点的白噪声  omega^2/Hz
Tphi1_Qidel_ZCZ3 = 4/WNoiseZPA_Qidel_ZCZ3  # 推算ZCZ3 bite 比特在idel点的T2

# %% 比特Tphi2推算
#EZQ1.1在低温33dB， 2.0e4码值的ZPA偏置下的P18_V3 比特 Tphi2 噪声 
# 包含片上噪声和DC源0偏置噪声
FNoiseZPA_Q01_V3 = 0.803e6 
Tphi2_Q01_V3 = 1/FNoiseZPA_Q01_V3

Tphi2_Qidel_ZCZ3 = Tphi2_Q01_V3*(ZMc_Q01_V3*kphiQ01_V3/ZMc_Q_ZCZ3/kphi_Qidel_ZCZ3)  # 推算ZCZ3 bite 比特在idel点的T2
print(Tphi2_Qidel_ZCZ3)
# %%计算固定电压噪声时的Tphi1
WNoiseZPA_V = 1.0e-17  #V^2/Hz 对应-163.3dBm/Hz
# WNoiseZPA_V = 1.3e-17  #V^2/Hz 对应-162dBm/Hz
T_gate_CZ = 40e-9
"""计算ZCZ3 coupler 在偏置300M处的T2大小"""
fmaxQ_ZCZ3 = 5.277e9
fahQ_ZCZ3 = -0.239e9
ZMc_Q_ZCZ3 = 1.85  # ZCZ3.1 qubit互感设计值 pH
fidel_ZCZ3 = fmaxQ_ZCZ3-300E6  # ZCZ3.1 idel点频率
phi_Qidel_ZCZ3, kphi_Qidel_ZCZ3 = flux_k_max_sim(
    fmaxQ_ZCZ3, fahQ_ZCZ3, fidel_ZCZ3)  # 计算ZCZ3 比特在idel点的斜率k_phi (HZ/phi_0)
# 推算ZCZ3 比特在idel点的白噪声  omega^2/Hz
WNoiseZPA_Qidel_ZCZ3 = WNoiseZPA_V * \
    (k_IVQ01_V3*ZMc_Q_ZCZ3)**2*kphi_Qidel_ZCZ3**2*(2*np.pi)**2
Tphi1_Qidel_ZCZ3 = 4/WNoiseZPA_Qidel_ZCZ3  # 推算ZCZ3 bite 比特在idel点的T2


"""计算ZCZ3 coupler 在打开20M 耦合处的T2大小"""
fmaxC_ZCZ3 = 10.5e9
f_C20M_ZCZ3 = 6.885e9  # ZCZ3.1 打开20M耦合频率
f_Coff_ZCZ3 = 9.31e9  # ZCZ3.1 Coupler关断点频率
fahC_ZCZ3 = -0.138e9
ZMc_C_ZCZ3 = 2.5  # ZCZ3.1 Coupler互感设计值 pH
# 计算ZCZ3 coupler 在打开20M 耦合处的斜率k_phi (HZ/phi_0)
phi_C20M_ZCZ3, kphi_C20M_ZCZ3 = flux_k_max_sim(
    fmaxC_ZCZ3, fahC_ZCZ3, f_C20M_ZCZ3)
phi_Coff_ZCZ3, kphi_Coff_ZCZ3 = flux_k_max_sim(
    fmaxC_ZCZ3, fahC_ZCZ3, f_Coff_ZCZ3)  # 计算ZCZ3 coupler 在关断点的斜率k_phi (HZ/phi_0)
# 推算ZCZ3 coupler 在打开20M 耦合处的白噪声  omega^2/Hz
WNoiseZPA_C20M_ZCZ3 = WNoiseZPA_V * \
    (k_IVQ10_V3*ZMc_C_ZCZ3)**2*kphi_C20M_ZCZ3**2*(2*np.pi)**2
# 推算ZCZ3 coupler 关断点的白噪声  omega^2/Hz
WNoiseZPA_Coff_ZCZ3 = WNoiseZPA_V * \
    (k_IVQ10_V3*ZMc_C_ZCZ3)**2*kphi_Coff_ZCZ3**2*(2*np.pi)**2
Tphi1_C20M_ZCZ3 = 4/WNoiseZPA_C20M_ZCZ3  # 推算ZCZ3 coupler 在打开20M 耦合处的T2
Tphi1_Coff_ZCZ3 = 4/WNoiseZPA_Coff_ZCZ3  # 推算ZCZ3 coupler 在关断点的T2


# leakage_C20M_ZCZ3 = 0.0216                #ZCZ3 比特在打开20M耦合处的在coupler上的态泄露
# leakage_Coff_ZCZ3 = 0.00615                    #ZCZ3 比特在关断点的在coupler上的态泄露
partialFreq_C20M_ZCZ3 = np.sqrt(0.0004382)  # ZCZ3 比特在打开20M耦合处频率随coupler变化的斜率
partialFreq_Coff_ZCZ3 = np.sqrt(1.09e-5)  # ZCZ3 比特在打开关断点处频率随coupler变化的斜率
partialXYg_Coff_ZCZ3 = np.sqrt(0.00043588)  # ZCZ3 比特在打开20M耦合处耦合强度随coupler变化的斜率

Tphi1_Q20Ms_ZCZ3 = Tphi1_C20M_ZCZ3 / \
    partialFreq_C20M_ZCZ3**2  # 推算单个coupler在打开20M耦合时引起的比特T2限制
Tphi1_Qoffs_ZCZ3 = Tphi1_Coff_ZCZ3 / \
    partialFreq_Coff_ZCZ3**2  # 推算单个coupler在关断时引起的比特T2限制
Tphi1_Q20M_ZCZ3 = 4/(WNoiseZPA_C20M_ZCZ3*partialFreq_C20M_ZCZ3**2
                     + 3*WNoiseZPA_Coff_ZCZ3*partialFreq_Coff_ZCZ3**2)  # 一个打开耦合20M，三个关断引起的比特T2限制
Tphi1_Qoff_ZCZ3 = 4/(4*WNoiseZPA_Coff_ZCZ3 *
                     partialFreq_Coff_ZCZ3**2)  # 四个关断引起的比特T2限制

Tphi1_20M_allQ = 1/(1/Tphi1_Q20M_ZCZ3+1/Tphi1_Qidel_ZCZ3)

error_Tphi1_CZ = np.exp(0.5*(
    -2*(T_gate_CZ/Tphi1_Qidel_ZCZ3)  # 两个比特Tphi1引起的错误
    - (2+np.sqrt(2))**2*T_gate_CZ/Tphi1_Q20Ms_ZCZ3
    # coupler打开20M引起的错误，由于是相干的故为(2+sqrt(2))**2, sqrt(2)来自耦合强度波动，由于斜率和频率斜率接近约为Tphi1_Q20Ms_ZCZ3
    - 3*(2+np.sqrt(2))**2*T_gate_CZ/Tphi1_Qoffs_ZCZ3 
))

## 到达比特前端电压噪声谱：
WNoiseV_Qidel_chip = WNoiseZPA_V*10**(-33/10)

dBmV_Qidel_chip = Svv_to_dBm(WNoiseZPA_V*10**(-33/10),50)

#%% 通过互感和Tphi1需求直接反推处理器前端电压噪声谱
e = 1.602e-19
h = 6.626e-34
phi_0 = h/2/e


fmaxQ_ZCZ3 = 5.277e9
fahQ_ZCZ3 = -0.239e9
ZMc_Q_ZCZ3 = 1.85  # ZCZ3.1 qubit互感设计值 pH
fidel_ZCZ3 = fmaxQ_ZCZ3-300E6  # ZCZ3.1 idel点频率
phi_Qidel_ZCZ3, kphi_Qidel_ZCZ3 = flux_k_max_sim(
    fmaxQ_ZCZ3, fahQ_ZCZ3, fidel_ZCZ3)  # 计算ZCZ3 比特在idel点的斜率k_phi (HZ/phi_0)
# 推算ZCZ3 比特在idel点的白噪声  omega^2/Hz

Tphi1_need = 1000e-6
WNoise_Qidel_need = 4/Tphi1_need  #omega^2/Hz
WNoisePhi_Qidel_need = WNoise_Qidel_need/(kphi_Qidel_ZCZ3**2*(2*np.pi)**2)*phi_0**2  # phi^2/Hz
WNoiseI_Qidel_need = WNoisePhi_Qidel_need/(ZMc_Q_ZCZ3*1e-12)**2   # I^2/Hz
R = 50
WNoiseV_Qidel_need = WNoiseI_Qidel_need*(R/2)**2   # V^2/Hz
dBmV_Qidel_need = Svv_to_dBm(WNoiseV_Qidel_need,50)