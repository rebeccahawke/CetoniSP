"""
This script aims to plot the response of a damped harmonic oscillator to a sinusoidal driving force
Note that this model uses the displacement as the starting point rather than the acceleration
"""

import numpy as np
import matplotlib.pyplot as plt

msl_orange = '#f15a29'

path = r'I:\MSL\Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\Models'

def d_n(t, n, A0, omega):
    return A0 * np.sin(n*omega*t)

def d(t, A0, omega):
    d = np.zeros(t_final*num_pts+1)
    for x in range(5000):
        n = 2*x + 1     # n odd
        d += d_n(t, n, A0, omega)
    return d

def v_n(t, n, A0, omega):
    return A0 * n * omega * np.cos(n*omega*t)

def v(t, A0, omega):
    v = np.zeros(t_final*num_pts+1)
    for x in range(5000):
        n = 2*x + 1     # n odd
        v += v_n(t, n, A0, omega)
    return v

def a_n(t, n, A0, omega, ):
    return -A0* n**2 * omega**2 * np.sin(n*omega*t)

def a(t, A0, omega):
    a = np.zeros(t_final*num_pts+1)
    for x in range(5000):
        n = 2*x + 1                 # n odd
        a += a_n(t, n, A0, omega, )
    return a


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