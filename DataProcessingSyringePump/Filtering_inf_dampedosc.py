import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear(x, m, c):
     return m * x + c

msl_orange = '#f15a29'

path = r'C:\Users\r.hawke\Desktop\4OctFiles\SyringePumpTests\20191002\Igor_Pro_data'
savepath = r'C:\Users\r.hawke\Desktop\4OctFiles\Working folder for APMF paper'

filename = '20191002_set1_r25sccm_infuse1_alldata.csv'
data_inf = pd.read_csv(path + '\\' + '20191002_set1_r25sccm_infuse1_alldata.csv')

t_params = pd.read_csv(path+'\\'+'t_params.csv', index_col='ID')

file_list = filename.split("_")
set = file_list[1]
flowrate = file_list[2].strip('r').strip('sccm')
run = file_list[3]
ID = set + flowrate + run

df = pd.read_csv(path+'\\'+filename)

time = df["time_sec"]
height = -1 * df["LVDT_volts"]

timestep = 0.1667
t_sub = 23.85
t_start = 11
a = t_sub + t_start
b = a + 10

pointA = int(a / timestep)
pointB = int(b / timestep)

plt.figure(figsize=(4,2))
#
data = plt.plot(time - t_sub, height,
            label="Data", marker='.', color=msl_orange, linestyle='None')

plt.xlim(t_start, t_start+10)
plt.ylim(5.1, 5.3)
plt.xlabel('Time (s)')
plt.ylabel('Piston position (mm)')
plt.tight_layout()


def dampedHO(t, A, omega0, Q, psi):
    gamma = omega0 / Q
    omega = np.sqrt(omega0 ** 2 - 0.5 * gamma ** 2)
    return A * np.exp(-gamma / 2 * t) * np.sin(omega * (t - psi))

time_synth = np.linspace(0, t_start+10, int(100*(t_start+10)+1))

offset = 5.202 - 0.0013*time_synth

t = time[pointA:pointB] - a
offset2 = 5.202 - 0.0013*t
h = height[pointA:pointB] - offset2

optParams, pcov = curve_fit(dampedHO, t, h)
print('Amplitude =', optParams[0])
print('Omega0 =', optParams[1])
print('Q =', optParams[2])
print('Phase =', optParams[3])

fit_osc = dampedHO(time_synth, optParams[0], optParams[1], optParams[2], optParams[3])
fit = plt.plot(time_synth + t_start, fit_osc + offset,
            label='Model', color='k', alpha=0.5)

plt.legend()

plt.savefig(savepath + '\\' + '20191002_set1_r25sccm_infuse1_osc_model.svg')
plt.savefig(savepath + '\\' + '20191002_set1_r25sccm_infuse1_osc_model.png', dpi=1200)

plt.show()
