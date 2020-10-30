import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate

path = r'I:\MSL\Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\Damped oscillator response'

f = 1/3                                    # frequency in Hz
amp_vel = 2.

piston_area = 196.12                        # mm2
syringe_dia = 23.03                         # mm
syringe_area = np.pi * (syringe_dia/2)**2

mult = 0.001*60 # multiplier mm^3 to ccm

t = np.arange(0, 9, 0.001)

### Sinusoidal functions
pos = amp_vel *(1 - np.cos(2 * np.pi * f * t))/(2*np.pi*f)  # position in mm
vel = amp_vel * np.sin(2*np.pi*f * t)     # velocity in mm/s

### Truncated trapezoid function
def trapezoid(t, width=1/f, slope=2., amp=2*amp_vel, phase=0):
    y = slope*width*signal.sawtooth(2*np.pi*t/width-phase, 0.5)
    y[y > amp / 2.] = amp / 2.
    y[y < -amp / 2.] = -amp / 2.
    return y

slope1=1*f*amp_vel
slope2 = 2*f*amp_vel

vel_1 = trapezoid(t, slope=slope1, phase=-np.pi/2)
vel_2 = trapezoid(t, slope=slope2, phase=-np.pi/2)
vel_3 = trapezoid(t, slope=3)

pos_1 = integrate.cumtrapz(vel_1, t, initial=0)
pos_2 = integrate.cumtrapz(vel_2, t, initial=0)

### Plot position and velocity

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex = True)

ax0.set_title('Driven oscillator with frequency '+str(round(f, 3))+' Hz')

ax0.plot(t, vel*piston_area*mult, label='sinusoid, A='+str(round(amp_vel*piston_area*mult,2)))
ax0.plot(t, vel_1*piston_area*mult, label='slope of f*A') # 'slope of '+str(round(slope1, 3)))
ax0.plot(t, vel_2*piston_area*mult, label='slope of 2*f*A') #'slope of '+str(round(slope2, 3)))
# ax2.plot(t, vel_3, label='slope of 3')
ax0.set_ylabel('Syringe flow rate (ccm)')
ax0.set_ylim(-25, 25)
ax0.legend()

ax1.plot(t, vel, label='sinusoid, A='+str(amp_vel))
ax1.plot(t, vel_1, label='slope of f*A') # 'slope of '+str(round(slope1, 3)))
ax1.plot(t, vel_2, label='slope of 2*f*A') #'slope of '+str(round(slope2, 3)))
# ax2.plot(t, vel_3, label='slope of 3')
ax1.set_ylabel('Piston velocity (mm/s)')
ax1.legend()

ax2.plot(t, pos, label='sinusoid, A='+str(round(amp_vel/(2*np.pi*f),2)))
ax2.plot(t, pos_1, label='slope of f*A')
ax2.plot(t, pos_2, label='slope of 2*f*A')
# ax2.plot(t, vel_3, label='slope of 3')
ax2.set_ylabel('Position (mm)')
ax2.set_xlabel('Time (s)')
ax2.legend()

plt.show()

#plt.savefig(path+'\\'+'Trapezoid_triangle_sinusoid.png')
