import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit #as opt
# from scipy.fftpack import fft
# from scipy.signal import blackman

def linear(x, m, c):
     return m * x + c

# timeint = 0.05
path = r'C:\Users\r.hawke\Desktop\4OctFiles\SyringePumpTests\20191002_fit2'
# path = r'C:\Users\r.hawke\Desktop\4OctFiles\SyringePumpTests\20191004'

filename = '20191002_step_25sccm_i_25.0_1569978651_alldata.csv'
# data_inf = pd.read_csv(path + '\\' + '20191002_step_25sccm_i_25.0_1569978651_alldata.csv')
# data_wdr = pd.read_csv(path + '\\' + '20191002_step_25sccm_w_25.0_1569978536_alldata.csv')

t_params = pd.read_csv(path+'\\'+'t_params.csv', index_col='Timestamp')


file_list = filename.split("_")
set = file_list[1]
flowrate = file_list[2].strip('r').strip('sccm')
run = file_list[3]
ID = int(file_list[-2]) #set + flowrate + run
# print(ID, t_params.loc[ID, 't1'])

df = pd.read_csv(path+'\\'+filename)

time = df["Time (s)"]
#height = df["syringe_mm"]
height = df["Height_rfc"]

t1 = int(t_params.loc[ID, 't1'])
p1 = int(t1/0.05)
# p1 = i1 - 10
t2 = int(t_params.loc[ID, 't2'])
p2 = int(t2/0.05)
t3 = int(t_params.loc[ID, 't3'])
p3 = int(t3/0.05) + 2

### Fit a sinusoidal function
# offset = 0
# t = time[:p1]
# V = height[:p1] + 0.002*time[:p1] - offset
#
# A_g = 0.04 # input('Estimated amplitude: ')
# f_g = 1.03 # input('Estimated frequency: ')
# phi_g = 0 # input('Estimated phase: ')
#
# a_guess = [A_g,f_g,phi_g]

offset = 9.485
t = time[p3:]
V = height[p3:] + 0.002*time[p3:] - offset

A_g = 0.03  # input('Estimated amplitude: ')
f_g = 1.02  # input('Estimated frequency: ')
phi_g = np.pi/2  # input('Estimated phase: ')

a_guess = [A_g, f_g, phi_g]

## Do the fit and decode the output


def sinusoid(t, A, f, phi):
    return A*np.cos(2*np.pi*f*t - phi)


a_fit, cov = curve_fit(sinusoid, t, V, p0=a_guess)
A_fit = a_fit[0]
f_fit = a_fit[1]
phi_fit = a_fit[2]

sig_A = np.sqrt(cov[0][0])
sig_f = np.sqrt(cov[1][1])
sig_phi = np.sqrt(cov[2][2])

## Display the results
print('A_inf = %g +/- %g' % (A_fit,sig_A))
print('f = %g +/- %g' % (f_fit,sig_f))
print('phi = %g +/- %g' % (phi_fit,sig_phi))

# V_fit = fit_func(t,A_fit,f_fit,phi_fit)
# plt.plot(t,V)
# plt.plot(t,V_fit, label='filtered')

A1 = -0.0334413
f1 = 1.05485
phi1 = 8.90099

A3 = 0.0308766
f3 = 1.0135
phi3 = -2.33181

V_fit1 = sinusoid(time[:p2], A1, f1, phi1)
V_fit3 = sinusoid(time[p2:], A3, f3, phi3)
plt.plot(time, height,label="original")
plt.plot(time[:p2], height[:p2] - V_fit1, label='filtered')
plt.plot(time[p2:], height[p2:] - V_fit3, label='filtered')
plt.legend()
plt.show()


'''
### FFT of 'before'
N = p1
T = 0.05
x = np.linspace(0.0, N*T, N)
hf = fft(height[0:p1])

w = blackman(N)
ywf = fft(height[0:p1]*w)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

plt.semilogy(xf[1:N//2], 2.0/N * np.abs(hf[1:N//2]), '-b')
plt.semilogy(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]), '-r')
plt.legend(['FFT', 'FFT w. window'])
plt.grid()
plt.show()
plt.clf()


## FFT of 'after'
N = len(height) - p3
T = 0.05
x = np.linspace(0.0, N*T, N)
hf = fft(height[p3:])

w = blackman(N)
ywf = fft(height[p3:]*w)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

plt.semilogy(xf[1:N//2], 2.0/N * np.abs(hf[1:N//2]), '-b')
plt.semilogy(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]), '-r')
plt.legend(['FFT', 'FFT w. window'])
plt.grid()
plt.show()
plt.clf()

'''
