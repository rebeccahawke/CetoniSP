"""
This script aims to plot the response of a damped harmonic oscillator to a sinusoidal driving force
"""

import numpy as np
import matplotlib.pyplot as plt

msl_orange = '#f15a29'

path = r'I:\MSL\Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\Models'

def a_n(t, n, A0, omega, ):
    return A0/n * np.sin(n*omega*t)

def a(t, A0, omega):
    a = np.zeros(t_final*num_pts+1)
    for x in range(5000):
        n = 2*x + 1                 # n odd
        a += a_n(t, n, A0, omega, )
    return a

def v_n(t, n, A0, omega):
    return -A0/(n**2 * omega) * np.cos(n*omega*t)

def v(t, A0, omega):
    v = np.zeros(t_final*num_pts+1)
    for x in range(5000):
        n = 2*x + 1     # n odd
        v += v_n(t, n, A0, omega)
    return v

def d_n(t, n, A0, omega):
    return -A0/(n**3 * omega**2) * np.sin(n*omega*t)

def d(t, A0, omega):
    d = np.zeros(t_final*num_pts+1)
    for x in range(5000):
        n = 2*x + 1     # n odd
        d += d_n(t, n, A0, omega)
    return d

def B_el_n(omega, n, A0, omega0, Qm):
    diff = omega0**2 - (n*omega)**2
    gamma_m = omega0 / (Qm)
    return A0/n * diff / (diff**2 + (gamma_m*n*omega)**2)

def B_inel_n(omega, n, A0, omega0, Qm):
    diff = omega0**2 - (n*omega)**2
    gamma_m = omega0 / (Qm)
    return -A0*omega*gamma_m / (diff**2 + (gamma_m*n*omega)**2)

def z_n(t, n, omega, A0, omega0, Qm):
    return B_el_n(omega, n, A0, omega0, Qm) * np.sin(n*omega*t) + B_inel_n(omega, n, A0, omega0, Qm) * np.cos(n*omega*t)

def z(t, omega, A0, omega0, Qm):
    z = np.zeros(t_final*num_pts+1)
    for x in range(5000):
        n = 2*x + 1     # n odd
        z += z_n(t, n+1, omega, A0, omega0, Qm)
    return z

"""Parameters"""
t_final = 200
num_pts = 100
t = np.linspace(0, t_final, int(num_pts * t_final + 1))

f0 = 1                      # natural frequency, which is around 1 Hz for current setup
omega0 = 2 * np.pi * f0

Q = 10                      # Quality factor - higher for low damping and longer oscillations
m = 2                       # Mass of floating elements, kg
Qm = Q * m
# omega1 = np.sqrt(omega0**2 - 0.5 * gamma**2)

f = f0*1/2                      # driving force frequency
omega = 2 * np.pi * f       # driving force angular frequency
A0 = 2 * f * 2 * np.pi      # maximum acceleration


"""Driving force behaviour"""
# fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True,)
#
# n=1
# ax1.plot(t, d_n(t, n, A0, omega), label="Driving")
# ax2.plot(t, z_n(t, n, omega, A0, omega0, Qm), label="Relative")
# ax3.plot(t, d_n(t, n, A0, omega) + z_n(t, n, omega, A0, omega0, Qm), label="Response")
# ax3.set_xlabel("Time (s)")
# ax2.set_ylabel("Displacement")
# figname = 'TheoreticalDrivingOscillation_3f=f0'

n_col = 10
color=plt.cm.rainbow(np.linspace(0,1,n_col))


"""Pure sinusoid with only one frequency  # # Driving with one sinusoid, n=1"""
# plt.plot(t, np.sin(omega*t))
# for Q,c in zip(range(1, n_col+1, 1),color):
#     plt.plot(t, z_n(t, 1, omega, A0, omega0, Q, m), label="Q="+str(Q), c=c)


"""Square wave driving force  # # Driving with square wave acceleration (many n=odd sinusoids)"""
# plt.plot(t, 0.0015*a(t, A0, omega), color='k', alpha=0.5)
#
"""varying Q"""
# for Q,c in zip(range(1, n_col+1, 1),color):
#     Qm = Q * m
#     plt.plot(t, z(t, omega, A0, omega0, Qm), label="Q="+str(Q), c=c)
# plt.xlabel("Time (s)")
# plt.ylabel("Displacement (mm)")
# figname="OscShape_one-f0"

"""varying omega, looking at max amplitude of oscillation"""
N = 100
n=1
relfreq = np.logspace(-2, 2, N, endpoint=True)
maxA = []
for x in relfreq:
    print(x)
    omega = omega0 * x
    maxA.append(np.sqrt(B_el_n(omega, n, A0, omega0, Qm))**2 + (B_inel_n(omega, n, A0, omega0, Qm))**2)
    # plt.plot(t, z(t, omega, A0, omega0, Q), label="omega="+str(x/(0.5*n_col)), c=c)

plt.loglog(relfreq, maxA, color=msl_orange)
plt.xlabel("Relative frequency, $\omega/\omega_0$")
plt.ylabel("Relative amplitude of oscillation, $A/A_0$")
# figname = "RelOscAmplitude_vs_RelFreq_LogLog_Sinusoid"


"""selected omegas"""
# n=1
# for x,c in zip(range(0, n_col+1, 1),color):
#     omega = omega0 * (2*(x+1)/n_col) #+ (x-(n_col+1)/2)/n_col)
#     plt.plot(t, (d_n(t, 1, A0, omega) + z_n(t, 1, omega, A0, omega0, Qm))/(A0/(n**3 * omega**2)), label="omega/omega0="+str(round(omega/omega0,3)), c=c)
#
# plt.xlim(0,10)
# plt.legend()
#
# plt.savefig(path + '\\' + figname + '.png', dpi=1200)
# plt.savefig(path + '\\' + figname + '.svg')

plt.show()
