import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear(x, m, c):
    return m * x + c

def sinusoid(t, A, f, phi):
    return A * np.cos(2 * np.pi * f * t - phi)

def dampedHO(t, A, omega0, Q, psi):
    gamma = omega0 / Q
    omega = np.sqrt(omega0 ** 2 - 0.5 * gamma ** 2)
    return A * np.exp(-gamma / 2 * t) * np.sin(omega * (t - psi))

msl_orange = '#f15a29'

path = r'C:\Users\r.hawke\Desktop\4OctFiles\SyringePumpTests\20191002_fit2'
# path = r'C:\Users\r.hawke\Desktop\4OctFiles\SyringePumpTests\20191004'

data_inf = pd.read_csv(path + '\\' + '20191002_step_25sccm_i_25.0_1569978651_alldata.csv')
data_wdr = pd.read_csv(path + '\\' + '20191002_step_25sccm_w_25.0_1569978536_alldata.csv')

t_params = pd.read_csv(path+'\\'+'t_params.csv', index_col='Timestamp')

timeint = 0.05

time_inf = data_inf['Time (s)']
height_inf = data_inf['Height_rfc']

t1_inf = 29.7
t2_inf = 30.7
t_sub = 19.7
t_sub2 = 20.6
i1_inf = int(t1_inf / timeint)
p2_inf = int(t2_inf / timeint)
p0_inf = i1_inf - int(10 / timeint)
p3_inf = p2_inf + int(20 / timeint)

linParamsT_inf, pcovT_inf = curve_fit(linear, time_inf[i1_inf:p2_inf],
                                          height_inf[i1_inf:p2_inf])  # piston travel rate

# filtering of noise - infusion
A1 = -0.0334413
f1 = 1.05485
phi1 = 8.90099

A3 = 0.0308766
f3 = 1.0135
phi3 = -2.33181

V_fit1 = sinusoid(time_inf[:p2_inf], A1, f1, phi1)
V_fit3 = sinusoid(time_inf[p2_inf:], A3, f3, phi3)

time_res_inf = time_inf[i1_inf:p2_inf] - t_sub
res_inf = height_inf[i1_inf:p2_inf] - linear(time_inf[i1_inf:p2_inf] - V_fit1[i1_inf:p2_inf], *linParamsT_inf)


# curve fitting of rates - withdrawal
time_wdr = data_wdr['Time (s)']
height_wdr = data_wdr['Height_rfc']

t1_wdr = 30.6
t2_wdr = 31.95
i1_wdr = int(t1_wdr / timeint)
p2_wdr = int(t2_wdr / timeint)
p0_wdr = i1_wdr - int(10 / timeint)
p3_wdr = p2_wdr + int(20 / timeint)

linParamsT_wdr, pcovT_wdr = curve_fit(linear, time_wdr[i1_wdr:p2_wdr],
                                          height_wdr[i1_wdr:p2_wdr])  # piston travel rate

# filtering of noise - withdrawal
A2 = 0.0297139
f2 = 1.20941
phi2 = 0.000846732

A4 = 0.0332837
f4 = 1.16115
phi4 = 3.30832

V_fit2 = sinusoid(time_inf, A2, f2, phi2)
V_fit4 = sinusoid(time_inf, A4, f4, phi4)

time_res_wdr = time_wdr[i1_wdr:p2_wdr] - t_sub
res_wdr = height_wdr[i1_wdr:p2_wdr] - linear(time_wdr[i1_wdr:p2_wdr] - V_fit2[i1_wdr:p2_wdr], *linParamsT_wdr)

# fit a damped harmonic oscillator

Q = 9
f_inf = 1.4
omega0_inf = 2 * np.pi * f_inf

A_inf = 20
psi_inf = +0.3*np.pi

# fit_res_inf = dampedHO(time_res_inf, A_inf, omega0_inf, Q, psi_inf)
#
# plt.plot(time_res_inf, res_inf,
#                 label="Data", marker='.', color=msl_orange, linestyle='None')
# plt.plot(time_res_inf, fit_res_inf)

Q = 4
f_wdr = 1.3
omega0_wdr = 2 * np.pi * f_wdr

A_wdr = 13000
psi_wdr = +0.31*np.pi

fit_res_wdr = dampedHO(time_res_wdr, A_wdr, omega0_wdr, Q, psi_wdr)

plt.plot(time_res_wdr, res_wdr,
                label="Data", marker='.', color=msl_orange, linestyle='None')
plt.plot(time_res_wdr, fit_res_wdr)

plt.show()