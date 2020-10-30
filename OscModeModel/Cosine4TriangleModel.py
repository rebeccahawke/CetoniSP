"""
This script aims to plot the response of a damped harmonic oscillator to a periodic driving force
"""

import numpy as np
import matplotlib.pyplot as plt

msl_orange = '#f15a29'

h = 3
w = 4

path = r'I:\MSL\Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\Models\DeltaDrivingForce_TriangleZ'
datapath = r'C:\Users\r.hawke\Desktop\4OctFiles\SyringePumpTests\20190925'

"""Driving (z_d) functions"""
max_n = 1000

def ad_n(t, n, A0, omega):
    return -A0*omega**2 * np.cos(n*omega*t)

def ad(t, A0, omega):
    a = np.zeros(len(t))
    for x in range(max_n):
        n = 2*x + 1                 # n odd
        a += ad_n(t, n, A0, omega)
    return a

def vd_n(t, n, A0, omega):
    return -A0*omega/n * np.sin(n*omega*t)

def vd(t, A0, omega):
    v = np.zeros(len(t))
    for x in range(max_n):
        n = 2*x + 1     # n odd
        v += vd_n(t, n, A0, omega)
    return v

def zd_n(t, n, A0, omega):
    return A0/n**2 * np.cos(n*omega*t)

def zd(t, A0, omega):
    d = np.zeros(len(t))
    for x in range(max_n):
        n = 2*x + 1     # n odd
        d += zd_n(t, n, A0, omega)
    return d

"""Response (z_r) functions"""
def A_el_n(n, A0, omega, omega0, Qm):
    diff = omega0**2 - (n*omega)**2
    c_m = omega0 / (Qm)
    return A0 * omega**2 * diff / (diff**2 + (c_m*n*omega)**2)

def A_inel_n(n, A0, omega, omega0, Qm):
    diff = omega0**2 - (n*omega)**2
    c_m = omega0 / (Qm)
    return A0*c_m*n*omega**3 / (diff**2 + (c_m*n*omega)**2)

def zr_n(t, n, A0, omega, omega0, Qm):
    return A_el_n(n, A0, omega, omega0, Qm) * np.cos(n * omega * t) + A_inel_n(n, A0, omega, omega0, Qm) * np.sin(n * omega * t)

def zr(t, A0, omega, omega0, Qm):
    z = np.zeros(len(t))
    for x in range(max_n):
        n = 2*x + 1     # n odd
        z += zr_n(t, n, A0, omega, omega0, Qm)
    return z

"""Measured position"""
def z_n(t, n, A0, omega, omega0, Qm):
    return zd_n(t, n, A0, omega, ) + zr_n(t, n, A0, omega, omega0, Qm)

def z(t, A0, omega, omega0, Qm):
    z = np.zeros(len(t))
    for x in range(max_n):
        n = 2*x + 1     # n odd
        z += z_n(t, n, A0, omega, omega0, Qm)
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

f = f0 * 1/10                    # driving force frequency
omega = 2 * np.pi * f       # driving force angular frequency
A0 = 1

"""Driving force behaviour"""
# fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True,)
#
# ax1.plot(t, zd(t, A0, omega), label="Displacement")
# ax2.plot(t, vd(t, A0, omega), label="Velocity")
# ax3.plot(t, ad(t, A0, omega), label="Acceleration")
# ax3.set_xlabel("Time (s)")
# # ax2.set_ylabel("Displacement")
# # figname = 'TheoreticalDrivingOscillation_3f=f0'
#
# plt.show()


#
# fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True,)
#
# z_d = zd(t, A0, omega)
# z_r = zr(t, A0, omega, omega0, Qm)
# z_t = z(t, A0, omega, omega0, Qm)
#
# ax1.plot(t, z_d, label="Driving")
# ax2.plot(t, z_r, label="Relative")
# ax3.plot(t, z_t, label="Response")
# ax3.set_xlabel("Time (s)")
# ax2.set_ylabel("Displacement")
# # figname = 'TheoreticalDrivingOscillation_3f=f0'
#
# plt.show()

"""varying Q"""
# for Q,c in zip(range(1, n_col+1, 1),color):
#     Qm = Q * m
#     plt.plot(time[612:2402], z(time[612:2402], psi, omega, A0, omega0, Qm), label="Q="+str(Q), c=c)

# plt.xlabel("Time (s)")
# plt.ylabel("Displacement (mm)")
# figname="OscShape_one-f0"

"""varying omega, looking at max amplitude of oscillation"""
# N = 1000
# relfreq = np.logspace(-1, 1, N, endpoint=True)
# maxT = []
# maxC = []
# for x in relfreq:
#     print(x)
#     omega = omega0 * x
#     # z_d = zd(t, A0, omega )
#
#     # z_t = z(t, A0, omega, omega0, Qm)
#     z_d_n = zd_n(t, 1, A0, omega)
#     z_t_n = z_n(t, 1, A0, omega, omega0, Qm)
#     z_r_n = zr_n(t, 1, A0, omega, omega0, Qm)
#     maxC.append(max(z_r_n) / max(z_d_n))
#     maxT.append(max(z_t_n) / max(z_d_n))
#
#     # plt.plot(t, z(t, omega, A0, omega0, Q), label="omega="+str(x/(0.5*n_col)), c=c)
# fig = plt.figure(figsize=(w,h*0.8))
# plt.loglog(relfreq, maxT, color=msl_orange, label="$z$")
# plt.loglog(relfreq, maxC, color='k', linestyle='dashdot', alpha=0.5, label='$z_r$')
# plt.xlabel("Relative frequency, $\omega/\omega_0$")
# plt.ylabel("Relative amplitude \nof oscillation, $A/A_0$")
# figname = "RelOscAmplitude_vs_RelFreq_LogLog_CosineOnly"
#
# plt.legend()
# plt.tight_layout(.5)
# #
# fig.savefig(path + '\\' + figname + '.png', dpi=1200)
# fig.savefig(path + '\\' + figname + '.svg')
#
# plt.show()
#
# plt.clf()
#

"""Comparison at natural frequency of triangle with pure cosine"""
# t_final = 3
# num_pts = 100
# t = np.linspace(0, t_final, int(num_pts * t_final + 1))
#
# fig = plt.figure(figsize=(w,h))
# plt.plot(t, z(t, A0, omega0, omega0, Qm)/max(zd(t, A0, omega )),
#          label='Triangle', color=msl_orange, )
# # plt.plot(syringe_flo, linear(syringe_flo, *linParamsLVDT), label="Fit LVDT")
# plt.plot(t, z_n(t, 1, A0, omega0, omega0, Qm)/max(zd(t, A0, omega )),
#          color='k', linestyle='dashdot', label='Cosine')
# plt.plot(t, ((z(t, A0, omega0, omega0, Qm) - z_n(t, 1, A0, omega0, omega0, Qm)))/max(zd(t, A0, omega )),
#          color='k', alpha=0.5, label='Difference')
#
# plt.legend(loc='lower left')
#
# plt.xlabel('Time (s)')
# plt.ylabel('Relative displacement',)
# plt.tight_layout(.5)
#
# figname2 = 'displacement_tri-v-cos_wdiff'
# fig.savefig(path + '\\' + figname2 + '.png', dpi=1200)
# fig.savefig(path + '\\' + figname2 + '.svg')
#
# plt.show()

"""Difference at natural frequency of triangle with pure cosine"""
# t_final = 3
# num_pts = 100
# t = np.linspace(0, t_final, int(num_pts * t_final + 1))
#
# fig = plt.figure(figsize=(3,1.5))
# plt.plot(t, ((z(t, A0, omega0, omega0, Qm) - z_n(t, 1, A0, omega0, omega0, Qm)))/max(zd(t, A0, omega )),
#          color='k', alpha=0.5, label='Difference')
#
# plt.xlabel('Time (s)')
# plt.ylabel('Relative \ndisplacement')
# plt.tight_layout(.5)
#
# figname2 = 'displacement_tri-v-cos_diff'
# fig.savefig(path + '\\' + figname2 + '.png', dpi=1200)
# fig.savefig(path + '\\' + figname2 + '.svg')
#
# plt.show()

"""selected omegas"""
n_col = 4
color=plt.cm.rainbow(np.linspace(0,1,n_col))
Qm = 5

t_final = 20
num_pts = 100
t = np.linspace(0, t_final, int(num_pts * t_final + 1))

for x,c in zip([1, 3, 6, 10],color):
    omega = omega0 * x #/(2**x) #+ (x-(n_col+1)/2)/n_col)
    plt.plot(t, z(t, A0, omega, omega0, Qm), label="w/w0 = "+str(x), c=c)

plt.legend()
figname='z_Ts_1Hz-f0_Triangle'
#
# plt.savefig(path + '\\' + figname + '.png', dpi=1200)
# plt.savefig(path + '\\' + figname + '.svg')
#
plt.show()

