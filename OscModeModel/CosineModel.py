"""
This script aims to plot the response of a damped harmonic oscillator to a periodic driving force
"""

import numpy as np
import matplotlib.pyplot as plt


msl_orange = '#f15a29'

path = r'I:\MSL\Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\Models'

"""Driving (z_d) functions"""
def zd_n(t, n, A0, omega):
    return A0 * np.cos(n*omega*t)

def zd(t, A0, omega):
    d = np.zeros(len(t))
    for x in range(5000):
        n = 2*x + 1     # n odd
        d += zd_n(t, n, A0, omega)
    return d

def vd_n(t, n, A0, omega):
    return -A0*n*omega * np.sin(n*omega*t)

def vd(t, A0, omega):
    v = np.zeros(len(t))
    for x in range(5000):
        n = 2*x + 1     # n odd
        v += vd_n(t, n, A0, omega)
    return v

def ad_n(t, n, A0, omega, ):
    return -A0*n**2*omega**2 * np.cos(n*omega*t)

def ad(t, A0, omega):
    a = np.zeros(len(t))
    for x in range(5000):
        n = 2*x + 1                 # n odd
        a += ad_n(t, n, A0, omega)
    return a

"""Response (z_r) functions"""
def A_el_n(omega, n, A0, omega0, Qm):
    diff = omega0**2 - (n*omega)**2
    c_m = omega0 / (Qm)
    return A0 * n**2 * omega**2 * diff / (diff**2 + (c_m*n*omega)**2)

def A_inel_n(omega, n, A0, omega0, Qm):
    diff = omega0**2 - (n*omega)**2
    c_m = omega0 / (Qm)
    return A0*c_m*n**3*omega**3 / (diff**2 + (c_m*n*omega)**2)

def zr_n(t, n, omega, A0, omega0, Qm):
    return A_el_n(omega, n, A0, omega0, Qm) * np.cos(n * omega * t) + A_inel_n(omega, n, A0, omega0, Qm) * np.sin(n * omega * t)

def zr(t, omega, A0, omega0, Qm):
    z = np.zeros(len(t))
    for x in range(5000):
        n = 2*x + 1     # n odd
        z += zr_n(t, n, omega, A0, omega0, Qm)
    return z

"""Measured position"""
def z_n(t, n, omega, A0, omega0, Qm):
    return zd_n(t, n, A0, omega, ) + zr_n(t, n, omega, A0, omega0, Qm)

def z(t, omega, A0, omega0, Qm):
    z = np.zeros(len(t))
    for x in range(5000):
        n = 2*x + 1     # n odd
        z += z_n(t, n, omega, A0, omega0, Qm)
    return z


"""Parameters"""
t_final = 60
num_pts = 100
t = np.linspace(0, t_final, int(num_pts * t_final + 1))

f0 = 1                      # natural frequency, which is around 1 Hz for current setup
omega0 = 2 * np.pi * f0

Q = 10                      # Quality factor - higher for low damping and longer oscillations
m = 2                       # Mass of floating elements, kg
Qm = Q * m
# omega1 = np.sqrt(omega0**2 - 0.5 * gamma**2)

f = f0 * 0.5                    # driving force frequency
omega = 2 * np.pi * f       # driving force angular frequency
A0 = 1.15

psi = 4

"""Driving force behaviour"""
# fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True,)
#
# z_d = zd_n(t, 1, A0, omega, )
# z_r = zr_n(t, 1, omega, A0, omega0, Qm)
#
# ax1.plot(t, z_d, label="Driving")
# ax2.plot(t, z_r, label="Relative")
# ax3.plot(t, z_d+z_r, label="Response")
# ax3.set_xlabel("Time (s)")
# ax2.set_ylabel("Displacement")
# # figname = 'TheoreticalDrivingOscillation_3f=f0'
#
# plt.show()


"""varying omega, looking at max amplitude of oscillation"""
N = 1000
relfreq = np.logspace(-2, 1, N, endpoint=True)
maxA = []
for x in relfreq:
    print(x)
    omega = omega0 * x
    z_d = zd_n(t, 1, A0, omega, )
    z_r = zr_n(t, 1, omega, A0, omega0, Qm)
    maxA.append(max(z_d+z_r)/max(z_d))

    # plt.plot(t, z(t, omega, A0, omega0, Q), label="omega="+str(x/(0.5*n_col)), c=c)

plt.loglog(relfreq, maxA, color=msl_orange)
plt.xlabel("Relative frequency, $\omega/\omega_0$")
plt.ylabel("Relative amplitude of oscillation, $A/A_0$")
figname = "RelOscAmplitude_vs_RelFreq_LogLog_SingleCosine"

plt.savefig(path + '\\' + figname + '.png', dpi=1200)
plt.savefig(path + '\\' + figname + '.svg')

plt.show()