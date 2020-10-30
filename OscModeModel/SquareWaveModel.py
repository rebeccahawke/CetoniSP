"""
This script aims to plot the response of a damped harmonic oscillator to a square wave driving force
"""

import numpy as np
import matplotlib.pyplot as plt

msl_orange = '#f15a29'

path = r'I:\MSL\Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\Models'

def a_n(t, n, omega, A0):
    return A0/n * np.sin(n*omega*t)

def a(t, A0, omega):
    a = np.zeros(t_final*num_pts+1)
    for x in range(5000):
        n = 2*x + 1                 # n odd
        a += a_n(t, n, omega, A0)
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

def zr(t, omega, A0, omega0, Qm):
    z = np.zeros(t_final*num_pts+1)
    for x in range(5000):
        n = 2*x + 1     # n odd
        z += z_n(t, n+1, omega, A0, omega0, Qm)
    return z

def z(t, omega, A0, omega0, Q):
    z = np.zeros(t_final*num_pts+1)
    for x in range(5000):
        n = 2*x + 1     # n odd
        z += z_n(t, n+1, omega, A0, omega0, Qm) + d_n(t, n, A0, omega)
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

f = f0/2                      # driving force frequency
omega = 2 * np.pi * f       # driving force angular frequency
A0 = 2 * f * 2 * np.pi      # maximum acceleration


"""Driving force behaviour"""
# plt.plot(t, a(t, A0, omega), label="Acceleration")
# plt.plot(t, v(t, A0, omega), label="Velocity")
# plt.plot(t, d(t, A0, omega), label="Displacement")
# plt.xlabel("Time (s)")
# plt.ylabel("Magnitude of driving oscillation")
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
relfreq = np.logspace(-2, 1, N, endpoint=True)
maxA = []
for x in relfreq:
    print(x)
    omega = omega0 * x
    z_d = d(t, A0, omega, )
    z_r = zr(t, omega, A0, omega0, Qm)
    maxA.append(max(z_d + z_r) / max(z_d))
    # plt.plot(t, z(t, omega, A0, omega0, Q), label="omega="+str(x/(0.5*n_col)), c=c)

plt.loglog(relfreq, maxA, color=msl_orange)
plt.xlabel("Relative frequency, $\omega/\omega_0$")
plt.ylabel("Relative amplitude of oscillation, $A/A_0$")
figname = "RelOscAmplitude_vs_RelFreq_LogLog_SquareAcc"

plt.savefig(path + '\\' + figname + '.png', dpi=1200)
plt.savefig(path + '\\' + figname + '.svg')

plt.show()





"""selected omegas"""
for x,c in zip([18, 9, 6, 4.4, 3.6, 3],color):
    omega = omega0 * 1/x #/(2**x) #+ (x-(n_col+1)/2)/n_col)
    z_d = d(t, A0, omega, )
    z_r = zr(t, omega, A0, omega0, Qm)
    plt.plot(t, (z_d + z_r)/max(z_d), label="omega/omega0="+str(round(omega/omega0,3)), c=c)

figname = 'z_truefs_1Hz-f0_SquareAcc'

plt.legend()

plt.savefig(path + '\\' + figname + '.png', dpi=1200)
plt.savefig(path + '\\' + figname + '.svg')

plt.show()
