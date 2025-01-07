# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 21:11:48 2023

@author: 馒头你个史剃磅
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

N = [0.014101426,0.028254384,0.056267525,0.11215459,0.089239372,
     0.070873052,0.044769289,0.022391752,0.017841165,0.011195258,
     0.008898129,0.007040927,0.005574743,0.004450671,0.003522621,
     0.002790041,0.002008082,0.035540162,0.001828113,3.03112E-05]
Gamma_phi = [51360.70852,86716.84443,153414.9099,309285.8149,244073.0406,
             193701.0882,132856.7232,70861.08466,64859.60459,46830.98656,
             39297.92976,33378.95879,28805.12496,25307.47961,24229.71603,
             23690.11232,21268.65544,110767.3976,18087.16807,15848.41452]
def linear_func(x,A,B):
    return(A*x+B)

fit_f = optimize.curve_fit(linear_func,Gamma_phi,N,[3e-7,0.006])

Fitted_N = [linear_func(n,*fit_f[0]) for n in Gamma_phi]
plt.figure()
plt.scatter(Gamma_phi,N)
plt.plot(Gamma_phi,Fitted_N)
