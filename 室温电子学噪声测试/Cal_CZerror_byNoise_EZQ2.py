# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:27:30 2023

@author: mantoutou
"""
#%% package导入定义
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import interpolate, optimize
from scipy import integrate


class Cal_CZerror_byNoise:
    """
    通过传入的噪声谱数据计算低频噪声引起的CZ门错误率
    """
    
    def __init__(self,
                 Freqs,
                 PSDs_dBm,
                 AT_qz_all = 33, # 比特 Z线总衰减器大小 dB
                 AT_cz_all = 21, # coupler Z线总衰减器大小 dB
                 tau_list = np.linspace(0.02e-6, 30e-6,201),
                 is_psd_in_chip = False,
                 T_gate_CZ = 50,  #CZ门时长 ns
                 ):
        """
        Args:
            Freqs: 单边噪声谱频率   List or Array 单位 Hz     range：(10^-3, 50e6)Hz  
            PSDs_dBm:  单边噪声谱功率   List or Array 单位 dBm/Hz   (单边噪声谱=双边噪声谱*2)
        """
        self.Freqs = Freqs
        self.PSDs_dBm = PSDs_dBm
        # 使用zip将Freq和PSD列表合并成元组列表，然后按照Freq的值排序
        sorted_data = sorted(zip(Freqs, PSDs_dBm), key=lambda x: x[0])
    
        # 解压排序后的数据并更新Freq和PSD
        Freq_r, PSD_r = map(np.array, zip(*sorted_data))
        PSD_r = self.dBm_to_Svv(np.array(PSD_r),1)

        PSD_r = PSD_r*2 ####### NOTE!!!!!!!!!!!!!
        
        self.Freq_r = Freq_r
        self.PSD_r = PSD_r
        
        (self.func_inter, self.fit_1f_data, self.PSD_w, 
         self.transf_1f, self.transf_w) = self.noise_data_process()
        
        if is_psd_in_chip:
            self.kv_c_20M = 58997285148.46638/np.sqrt(10**(-21/10))  #partial(f)/partial(V)  coupler在20M耦合处的频率随电压变化的斜率
            self.kv_c_off = 29950204279.18776/np.sqrt(10**(-21/10))  #partial(f)/partial(V)  coupler在关断点处的频率随电压变化的斜率
            self.kv_q_300M = (3036969283.1584916*(1.5/1.85)
                              /np.sqrt(10**(-33/10))) #partial(f)/partial(V)  qubit 在偏置300M处的频率随电压变化的斜率
        else:
            self.kv_c_20M = (58997285148.46638
                             *np.sqrt(10**(-(AT_cz_all-21)/10)))#partial(f)/partial(V)  coupler在20M耦合处的频率随电压变化的斜率
            self.kv_c_off = (29950204279.18776
                             *np.sqrt(10**(-(AT_cz_all-21)/10)))#partial(f)/partial(V)  coupler在关断点处的频率随电压变化的斜率
            self.kv_q_300M = (3036969283.1584916*(1.5/1.85)
                              *np.sqrt(10**(-(AT_qz_all-33)/10)))#partial(f)/partial(V)  qubit 在偏置300M处的频率随电压变化的斜率
        self.kphi_Qidel_ZCZ3 = -4106170488.0404496      #比特偏置300M位置的磁通斜率   partial(f)/partial(phi_0)
        self.kphi_Coff_ZCZ3 = -11564836971.289932
        self.kphi_C20M_ZCZ3 = -22780945937.148113
        self.partialFreq_C20M_ZCZ3 = np.sqrt(0.0004382)  # 比特在打开20M耦合处频率随coupler变化的斜率
        self.partialFreq_Coff_ZCZ3 = np.sqrt(1.09e-5)  # 比特在打开关断点处频率随coupler变化的斜率
        self.partialXYg_C20M_ZCZ3 = np.sqrt(0.00043588)  # 比特在打开20M耦合处耦合强度随coupler变化的斜率
        

        
        """计算T2*"""
        self.tau_list = tau_list
        Tphi1_q_in = 4/(self.PSD_w*(self.kv_q_300M*2*np.pi)**2)
        Tphi1_c20M_in = 4/(self.PSD_w*(2*np.pi*self.kv_c_20M*self.partialFreq_C20M_ZCZ3)**2)
        Tphi1_XYg_c20M_in = 4/(self.PSD_w*(2*np.pi*self.kv_c_20M*self.partialXYg_C20M_ZCZ3)**2)
        (PN_q,T2s_q,Tphi1_q ,
         Tphi2_q,fit_datas_q) = self.Cal_T2_by_Sw(Sw_func = self.PSD_omega_q300M,
                                                    omega_s = 0.01*2*np.pi,
                                                    omega_e = 50e6*2*np.pi,
                                                    int_limit = 5e3,
                                                    Tphi1 = Tphi1_q_in,
                                                    tau_list = self.tau_list,)
        (PN_20M,T2s_20M,Tphi1_20M ,
         Tphi2_20M,fit_datas_20M) = self.Cal_T2_by_Sw(Sw_func = self.PSD_omega_c20M,
                                                        omega_s = 0.01*2*np.pi,
                                                        omega_e = 50e6*2*np.pi,
                                                        int_limit = 5e3,
                                                        Tphi1 = Tphi1_c20M_in,
                                                        tau_list = self.tau_list,)
        # (PN_off,Interg_off,T2s_off,fit_datas_off) = self.Cal_T2_by_Sw(Sw_func = self.PSD_omega_coff,
        #                                                  omega_s = 0.005*2*np.pi,
        #                                                  omega_e = 50e6*2*np.pi,
        #                                                  int_limit = 5e3,
        #                                                  tau_list = np.linspace(0.02e-6, 80e-6, 201)
        #                                                  )
        (PN_geff,T2s_geff,Tphi1_geff ,
         Tphi2_geff,fit_datas_geff) = self.Cal_T2_by_Sw(Sw_func = self.PSD_XYg_c20M,
                                                         omega_s = 0.01*2*np.pi,
                                                         omega_e = 50e6*2*np.pi,
                                                         int_limit = 5e3,
                                                         Tphi1 = Tphi1_XYg_c20M_in,
                                                         tau_list = self.tau_list,)
        self.PN_q = PN_q
        self.T2s_q = T2s_q
        self.Tphi1_q = Tphi1_q
        self.Tphi2_q = Tphi2_q
        self.fit_datas_q = fit_datas_q
        self.PN_20M = PN_20M
        self.T2s_20M = T2s_20M
        self.Tphi1_20M = Tphi1_20M
        self.Tphi2_20M = Tphi2_20M
        self.fit_datas_20M = fit_datas_20M
        # self.PN_off = PN_off
        # self.T2s_off = T2s_off
        # self.fit_datas_off = fit_datas_off
        self.PN_geff = PN_geff
        self.T2s_geff = T2s_geff
        self.Tphi1_geff = Tphi1_geff
        self.Tphi2_geff = Tphi2_geff
        self.fit_datas_geff = fit_datas_geff
        
        self.T_gate_CZ = T_gate_CZ    #CZ门时间
        self.error_Tphi2_CZ = 1-np.exp(
            (-2*(self.T_gate_CZ/self.Tphi2_q)**2
            - (2*self.T_gate_CZ/self.Tphi2_20M+np.sqrt(2)*self.T_gate_CZ/self.Tphi2_geff)**2
            )/2
        )
        self.error_Tphi1_CZ = 1-np.exp(
            (-2*(self.T_gate_CZ/self.Tphi1_q)
             -2*(self.T_gate_CZ/self.Tphi1_20M)-2*self.T_gate_CZ/self.Tphi1_geff
             )/2
            )
        Tphi1 = 1/(1/self.Tphi1_q+1/self.Tphi1_20M)
        Tphi2 = np.sqrt(1/((1/Tphi2_q)**2+(1/Tphi2_20M)**2))
        
        print(f"""
              =========================================
              Tphi1: {np.round(Tphi1/1e-6,2)}us, Tphi2: {np.round(Tphi2/1e-6,2)}us @g=20MHz; 
              The expected CZ-gate error by Tphi2 is: {np.round(self.error_Tphi2_CZ*100,4)}% ! 
              The expected CZ-gate error by Tphi1 is: {np.round(self.error_Tphi1_CZ*100,4)}% ! 
              The expected total CZ-gate error is: {np.round((self.error_Tphi1_CZ+self.error_Tphi2_CZ)*100,4)}% ! 
              The target error should to be controlled below 0.08%.
              =========================================
              """)

    def noise_data_process(self,):
        """
        将噪声谱数据转化为可用的噪声谱函数
        """
        
        """对原始数据进行smooth平均后再利用插值形成噪声谱函数形式"""
        PSD_r = self.PSD_r
        Freq_r = self.Freq_r
        PSD_r_s = self.smooth(PSD_r, 5)
        func_inter_log=interpolate.UnivariateSpline(np.log(Freq_r),
                                                    [np.log(psd) for psd in PSD_r_s],
                                                    k=5,s=31)
        func_inter = lambda f: np.exp(func_inter_log(np.log(f)))
        PSD_inter=func_inter(Freq_r)

        """对低于1HZ的数据进行拟合得到1/f噪声的拟合结果"""
        index_1f = np.where(np.array(Freq_r)<1)[0]
        amp_0 = ((PSD_r_s[index_1f][0] - PSD_r_s[index_1f][-1])/
                 (1/Freq_r[index_1f][0] - 1/Freq_r[index_1f][-1]))
        fit_1f_data,fit_1f_error = optimize.curve_fit(self.Fit_PSD_1f_log_func,Freq_r[index_1f],
                                                [np.log(psd) for psd in PSD_r_s[index_1f]], 
                                                p0 = [amp_0,1,PSD_r_s[index_1f][-1]], 
                                                # bounds = (0,np.inf),
                                                maxfev = 400000)
        PSD_1f_fit = np.exp(self.Fit_PSD_1f_log_func(Freq_r,*fit_1f_data))
    
        """对高于10MHZ的数据进行平均得到白噪声结果"""
        index_w = np.where(np.array(Freq_r)>10e6)[0]
        PSD_w = np.mean(PSD_r[index_w])
    
        """
        拟合数据和白噪声转折点，
        寻找符合区间内拟合值(或平均得到的白噪值)与插值数据最靠近的点作为转折点，
        可以避免曲线出现突然的变化
        """
        transf_1f = np.argmin(np.abs(np.log10(PSD_inter[index_1f])-np.log10(PSD_1f_fit[index_1f])))
        transf_w = np.argmin(np.abs(np.log10(PSD_inter)-np.log10(PSD_w)))
        
        return(func_inter,fit_1f_data, PSD_w, transf_1f, transf_w)
    def dBm_to_Svv(self,
                   P_s, # 噪声谱功率 dBm, P_s = V_n*I_n
                   B_w,  # 采样带宽 Hz
                  ):
            
        """ 定义将频谱仪数据转化为S_vv函数 """
        S_vv = (10**(P_s/10))*1e-3/B_w*4*50
        return(S_vv)
    
    def Svv_to_dBm(self,
                   P_s, # 噪声谱功率 V^2/Hz
                   R = 50,
                  ):
        """ 定义将V^2/Hz 数据转化为dBm/Hz  10*lg(Vn^2/4R/1mW)"""
        S_dBm = 10*np.log10(P_s/1e-3/(4*R))
        return(S_dBm)


    def smooth(self,x,num):
        """定义smooth函数"""
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
    
    def PSD_func(self,
                 f):
        """
        生成噪声谱函数，
        在低于1/f噪声转折点使用1/f拟合结果生成噪声谱数据，
        在高于白噪声转折点使用10MHz以上的平均数值生成噪声数据，
        在两个转折点之间则使用插值函数生成噪声数据。
        """
        if f < self.Freq_r[self.transf_1f]:
            psd = np.exp(self.Fit_PSD_1f_log_func(f,*self.fit_1f_data))
        elif f > self.Freq_r[self.transf_w]:
            psd = self.PSD_w
        else:
            psd = self.func_inter(f)
        return(psd)
    
    def Fit_PSD_1f_log_func(self,f,A,alpha,B):
        return(np.log(A/f**alpha+B))
    
    def filter_Sw_Ramsey(self, Sw_func,omega_noise,tau):
        """Eamsey滤波函数*噪声谱, omega_noise 单位HZ；tau 单位s"""
        g_r=np.sin(omega_noise*tau/2)**2/(omega_noise*tau/2)**2
        f_s = g_r*Sw_func(omega_noise)
        return (f_s)
    def PSD_XYg_c20M(self,omega):
        psd_geff = self.PSD_func(omega/2/np.pi)*(2*np.pi*self.kv_c_20M*self.partialXYg_C20M_ZCZ3)**2
        psd = psd_geff
        return psd/2/np.pi      #omega^2/omega
    def PSD_omega_c20M(self, omega):
        psd_20M = self.PSD_func(omega/2/np.pi)*(2*np.pi*self.kv_c_20M*self.partialFreq_C20M_ZCZ3)**2
        psd = psd_20M
        return psd/2/np.pi      #omega^2/omega
    def PSD_omega_coff(self,omega):
        psd_off = self.PSD_func(omega/2/np.pi)*(2*np.pi*self.kv_c_off*self.partialFreq_Coff_ZCZ3)**2
        psd = psd_off
        return psd/2/np.pi      #omega^2/omega
    def PSD_omega_q300M(self,omega):
        psd_q = self.PSD_func(omega/2/np.pi)*(self.kv_q_300M*2*np.pi)**2
        psd = psd_q
        return psd/2/np.pi      #omega^2/omega
    
    def fit_Tp_func(self,t,Tphi2,Tphi1): 
        """拟合T2降到1/e时的位置 (同时拟合高斯和指数)"""
        return( np.exp(-t**2/Tphi2**2-t/Tphi1))
    
    def Cal_T2_by_Sw(self,
                     Sw_func,
                     omega_s = 0.1*2*np.pi,
                     omega_e = 50e6*2*np.pi,
                     int_limit = 5e3,
                     fit_T2_guess = None,
                     Tphi1 = None,
                     tau_list = np.linspace(0.02e-6, 30e-6, 201)
                     ):
        P_N=[]
        Interg=[]
        for i in range(len(tau_list)): 
            
            tau=tau_list[i]
            """噪声谱分部积分 避免跨度太大积分报错 (每隔两个数量级进行积分)"""
            NMin = np.floor(np.log10(omega_s/2/np.pi))
            NMax = np.ceil(np.log10(omega_e/2/np.pi))
            NList = np.arange(NMin,NMax,2)
            
            Inerg_omega_nod = [10**n*2*np.pi for n in NList]
            Inerg_omega_nod[0] = omega_s
            Inerg_omega_nod[-1] = omega_e
            # print(Inerg_omega_nod)
            Int_func = lambda omega: self.filter_Sw_Ramsey(Sw_func,omega,tau)
            Int_S = 0
            for ii in range(len(Inerg_omega_nod)-1):
                Int_S_ii, err_ii = integrate.quad(
                    Int_func, Inerg_omega_nod[ii],Inerg_omega_nod[ii+1], 
                    limit=int(int_limit),
                    # epsabs = 1e-24,
                    # epsrel = 1e-24,
                    )
                Int_S += Int_S_ii
            # print(Int_S)
            # Int_D = Int_S
            Interg.append(Int_S)
            p=np.exp(-tau**2*Int_S/2)
            P_N.append(p)
        
        """利用拟合P下降到1/e时的T2"""
        if fit_T2_guess is None and Tphi1 is None:
            fit_T2_guess = [2*tau_list[np.argmin(abs(np.array(P_N)-1/np.e))],
                            2*tau_list[np.argmin(abs(np.array(P_N)-1/np.e))]]
        elif fit_T2_guess is None and Tphi1:
            fit_T2_guess = [2*tau_list[np.argmin(abs(np.array(P_N)-1/np.e))]]
        # print(fit_T2_guess)
        if Tphi1:
            fit_datas,fit_errors = optimize.curve_fit(
                lambda t,Tphi2:self.fit_Tp_func(t,Tphi2,Tphi1=Tphi1), 
                tau_list, P_N, fit_T2_guess,maxfev = 40000)
        else:
            fit_datas,fit_errors = optimize.curve_fit(
                self.fit_Tp_func, tau_list, P_N, fit_T2_guess,maxfev = 40000)
        fit_tphi1 = Tphi1 if Tphi1 else fit_datas[1]
        fit_tphi2 = fit_datas[0]
        a = (1/fit_tphi2)**2; b = 1/fit_tphi1; c=-1
        T2_e = (-b+np.sqrt(b**2-4*a*c))/(2*a)
        
        return(P_N,T2_e,fit_tphi1,fit_tphi2,fit_datas,)

#%%  Foe Example
starttime=int(time.time())

"""生成模拟噪声谱数据进行演示"""
Freq_r = np.linspace(1e-3**0.1, 50e6**0.1,5001)
Freq_r = Freq_r**10
PSD_Svv = [(7.9e-17)*(1+np.random.uniform(-0.15,0.15)) for f in Freq_r] ##单边噪声谱！！
# PSD_Svv = [(0.95e-11/f**1.1+1.2e-17)*(1+np.random.uniform(-0.15,0.15)) for f in Freq_r] ##单边噪声谱！！

PSD_dBm = [10*np.log10(P_s/1e-3/(4*50)) for P_s in PSD_Svv]
tau_list = np.linspace(0.02e-6, 30e-6, 201)

"""
输入室温噪声谱的频率和谱密度(dBm)，
要求单边噪声谱，频率范围最小要1mHz， 最大要超过50MHz
也可以输入芯片前端噪声谱数据，此时需将参数 is_psd_in_chip = True
"""
CZerror_Caled = Cal_CZerror_byNoise(Freq_r, PSD_dBm, 
                                    AT_qz_all=39.02,
                                    AT_cz_all=32.09,
                                    is_psd_in_chip = False,
                                    T_gate_CZ = 50,
                                    tau_list = tau_list)
t_list = CZerror_Caled.tau_list

"""绘制噪声谱"""
Freq_r = CZerror_Caled.Freq_r
PSD_r = CZerror_Caled.PSD_r
fig,axes=plt.subplots()
twin_axes=axes.twinx() 
axes.plot(Freq_r,PSD_r, label = 'Original PSD datas')
axes.plot(Freq_r,[CZerror_Caled.PSD_func(f) for f in Freq_r],
          label = 'Processed PSD datas')
axes.set_xscale('log')
axes.set_yscale('log')
y1, y2 = axes.get_ylim() 
axes.set_ylabel(r'$S_{vv}~ (V^2/Hz)$')
axes.set_xlabel('Freq (HZ)')
twin_axes.set_ylim(CZerror_Caled.Svv_to_dBm(y1),CZerror_Caled.Svv_to_dBm(y2))
twin_axes.set_ylabel('PSD (dBm/Hz)')
axes.legend()
plt.show()

"""绘制预测的T2数据"""
plt.figure()
plt.scatter(t_list/1e-6,CZerror_Caled.PN_q, s=1,
         label='Simulated dephasing of qubits bias 300M')
plt.plot(t_list/1e-6,CZerror_Caled.fit_Tp_func(t_list,*CZerror_Caled.fit_datas_q,CZerror_Caled.Tphi1_q), 'r--',
         label=rf'Fited Qubits-300M $T_2^{{1/e}} = {round(CZerror_Caled.T2s_q/1e-6,2)} \mu$s')
plt.scatter(t_list/1e-6,CZerror_Caled.PN_20M, s=1,
         label=r'Simulated dephasing of qubits due to $g_{eff}^{XY} = 20M$')
plt.plot(t_list/1e-6,CZerror_Caled.fit_Tp_func(t_list,*CZerror_Caled.fit_datas_20M,CZerror_Caled.Tphi1_20M),'b--',
         label=rf'Fited $g_{{eff}}^{{XY}} = 20M$ with qubits $T_2^{{1/e}} = {round(CZerror_Caled.T2s_20M/1e-6,2)} \mu$s')
plt.legend()

endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')