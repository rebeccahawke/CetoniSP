"""
This script plots the response of a damped harmonic oscillator to a periodic driving force
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from openpyxl import Workbook

from data_handling.process_files import get_all_fnames
from data_handling.plot_data import linear, fit_linear

msl_orange = '#f15a29'

path = r'I:\MSL\Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\Models\DeltaDrivingForce_TriangleZ'
datapath = r'C:\Users\r.hawke\Desktop\4OctFiles\SyringePumpTests\20190925'

"""Driving (z_d) functions"""
max_n = 100

def ad_n(t, n, A0, omega):
    return A0*omega * np.cos(n*omega*t)

def ad(t, A0, omega):
    a = np.zeros(len(t))
    for x in range(max_n):
        n = 2*x + 1                 # n odd
        a += ad_n(t, n, A0, omega)
    return a

def vd_n(t, n, A0, omega):
    return A0/n * np.sin(n*omega*t)

def vd(t, A0, omega):
    v = np.zeros(len(t))
    for x in range(max_n):
        n = 2*x + 1     # n odd
        v += vd_n(t, n, A0, omega)
    return v

def zd_n(t, n, A0, omega):
    return -A0/(n**2 * omega) * np.cos(n*omega*t)

def zd(t, A0, omega):
    d = np.zeros(len(t))
    for x in range(max_n):
        n = 2*x + 1     # n odd
        d += zd_n(t, n, A0, omega)
    return d

"""Response (z_r) functions"""
def A_el_n(n, A0, omega, omega0, Qm):
    diff = omega0**2 - (n*omega)**2
    c_m = omega0 / Qm
    return -A0 * omega * diff / (diff**2 + (c_m*n*omega)**2)

def A_inel_n(n, A0, omega, omega0, Qm):
    diff = omega0**2 - (n*omega)**2
    c_m = omega0 / Qm
    return -A0*omega*c_m*n*omega / (diff**2 + (c_m*n*omega)**2)

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

"""Driving force behaviour"""
def plot_response(t, A0, omega, omega0, Qm):
    # plot driving, relative and response displacements
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True,)

    z_d = zd(t, A0, omega)
    z_r = zr(t, A0, omega, omega0, Qm)

    ax1.plot(t, z_d, label="Driving")
    ax2.plot(t, z_r, label="Relative")
    ax3.plot(t, z_d+z_r, label="Response")  # equivalent to z(t, A0, omega, omega0, Qm)

    ax3.set_xlabel("Time (s)")
    ax2.set_ylabel("Displacement")

    for ax in [ax1, ax2, ax3]:
        ax.legend()
    # figname = 'TheoreticalDrivingOscillation_3f=f0'

    plt.show()


n_col = 6
color=plt.cm.rainbow(np.linspace(0,1,n_col))

"""varying Q"""
# for Q,c in zip(range(1, n_col+1, 1),color):
#     Qm = Q * m
#     plt.plot(time[612:2402], z(time[612:2402], psi, omega, A0, omega0, Qm), label="Q="+str(Q), c=c)

# plt.xlabel("Time (s)")
# plt.ylabel("Displacement (mm)")
# figname="OscShape_one-f0"

"""varying omega, looking at max amplitude of oscillation"""
def vary_omega(path=None, figname=None):
    # this function takes a minute to execute...
    N = 100
    relfreq = np.logspace(-2, 2, N, endpoint=True)
    maxR = []
    maxD = []
    for x in relfreq:
        print(x)
        omega = omega0 * x
        z_d = z(t, A0, omega, omega0, Qm)
        z_r = zr(t, A0, omega, omega0, Qm)
        maxR.append(max(z_r))
        maxD.append(max(z_d))

        # plt.plot(t, z(t, omega, A0, omega0, Q), label="omega="+str(x/(0.5*n_col)), c=c)

    plt.loglog(relfreq, maxR, color=msl_orange)
    plt.loglog(relfreq, maxD, color='k')
    plt.xlabel("Relative frequency, $\omega/\omega_0$")
    plt.ylabel("Relative amplitude of oscillation, $A/A_0$")
    figname = "RelOscAmplitude_vs_RelFreq_LogLog_Triangle_r&d"

    if path is not None and figname is not None:
        plt.savefig(os.path.join(path, figname + '.png'), dpi=1200)
        plt.savefig(os.path.join(path, figname + '.svg'))

    plt.show()
    plt.clf()


"""selected omegas"""
def plot_some_omegas(figname=None):
    for x,c in zip([10, 3, 2.5, 2, 1, 0.5],color):
        omega = omega0 * 1/x #/(2**x) #+ (x-(n_col+1)/2)/n_col)
        plt.plot(t, z(t, A0, omega, omega0, Qm), label="period (s) = "+str(x), c=c)

    plt.legend()
    if figname is None:
        figname='z_truefs_1Hz-f0_Triangle'

    plt.show()

"""Delta driving force  # # Driving with square wave velocity (many n=odd sinusoids)"""
# plt.plot(t, 0.0015*a(t, A0, omega), color='k', alpha=0.5)

# """5 sccm, 0.75 mL"""
# rawdata05 = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_5sccm_5.0_1569366092_alldata.csv')
# """10 sccm, 0.75 mL"""
# rawdata10 = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_10sccm_10.0_1569366539_alldata.csv')
# """15 sccm, 0.75 mL"""
# rawdata15 = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_15sccm_15.0_1569366892_alldata.csv')
# """20 sccm, 0.75 mL"""
# rawdata20 = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_20sccm_20.0_1569368175_alldata.csv')
# """25 sccm, 0.75 mL"""
# rawdata25 = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_25sccm_25.0_1569368549_alldata.csv')
# """30 sccm, 0.75 mL"""
# rawdata30 = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_30sccm_30.0_1569369842_alldata.csv')
#
#
# rawdata=rawdata10
# time = rawdata['Time (s)'] - rawdata['Time (s)'][0]
# height = rawdata['Height_LVDT']

# plt.plot(time[612:2402], height[612:2402], label="Raw data")
# plt.show()

# optParams, pcov1 = curve_fit(z, time[612:2402], height[612:2402])
# print(optParams)
# # -1.59770892,  0.98723277,  0.05248419,  9.54970133,  7.86007956
# optParams = [psi,  omega,  A0, omega0, Qm]
#
# # print('A0 =', optParams[1])
# # print('Omega =', optParams[0])
# # print('Omega0 =', optParams[2])
# # print('Qm =', optParams[3])
#
# # t, psi, omega, A0, omega0, Qm
#
#
# psi = optParams[0]
# omega = optParams[1]
# A0 = optParams[2]
# omega0 = optParams[3]
# Qm = optParams[4]
#
# # omega = omega0 * 1/9
# plt.plot(time, height, label="Raw data", c='k')
# """Varying omega0"""
# for x,c in zip(range(1, n_col+1, 1),color):
#     f0 = 0.5 + x / 20
#     omega0 = 2 * np.pi * f0
#     plt.plot(time[612:2402], z(time[612:2402], 0, omega, A0, omega0, Qm), label="Omega0=2pi"+str(f0), c=c)
#
# # plt.plot(t+42.1, z(t, omega, A0, 2 * np.pi *0.8, Qm)+0.3, label="Model z, $f_0 = 0.8$")
# # plt.plot(t+42.1, z(t, omega, A0, 2 * np.pi *0.85, Qm)+0.3, label="Model z, $f_0 = 0.85$")
# # plt.plot(t+42.1, z(t, omega, A0, 2 * np.pi *0.9, Qm)+0.3, label="Model z, $f_0 = 0.9$")
# plt.xlim(100,120)
# #
# # figname = '05sccm_0.75mL_f0-0.75-9Hz'
# plt.legend()
# #
# # plt.savefig(path + '\\' + figname + '.png', dpi=1200)
# # plt.savefig(path + '\\' + figname + '.svg')
# #
# plt.show()


"""Fit response function to real data"""
def sin_fit(t, A, f, phi):
    # f = driving force frequency as set by syringe pump
    omega = 2 * np.pi * f  # driving force angular frequency
    return A * np.sin(omega * t + phi)


def fit_response_single_sine(folder, filename, A0=1., f=1., ):
    # fit with single sinusoid

    num_pts = 1500
    t = np.linspace(0, num_pts*0.02, num=num_pts)

    path = os.path.join(folder, filename)
    df = pd.read_excel(path, sheet_name="RFC data")
    t0 = 2400

    # find best linear fit
    linpars = fit_linear(t, df["LVDT (mm)"][t0:t0+num_pts])
    # subtract linear fall rate
    z_real = df["LVDT (mm)"][t0:t0+num_pts] - linear(t, *linpars)
    # plt.plot(t, z_real)
    # plt.show()

    # Fit to a pure sinusoid
    pars, cov = curve_fit(f=sin_fit, xdata=t, ydata=z_real, p0=[A0, f, 0.1], bounds=(-np.inf, np.inf))
    # # Get the standard deviations of the parameters (square roots of the diagonal of the covariance)
    # stdevs = np.sqrt(np.diag(cov))
    # Calculate a sum square of the residuals as a measure of how 'clean' the sinusoid is:
    res = z_real - sin_fit(t, *pars)
    res_sum = sum(i * i for i in res)
    # print("Pars {} {} {} {}\nCov {}\nStdevs {}".format(*pars, cov, stdevs))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, )
    ax1.plot(t, z_real, label="Expt")

    ax2.plot(t, z_real, label="Expt")
    ax2.plot(t, sin_fit(t, *pars), label="Fit")

    ax3.plot(t, z_real - sin_fit(t, *pars), label='Residual sin')
    # ax3.plot(t, z_real - z_fit(t, *z_pars), label="Difference model")

    for ax in [ax1, ax2, ax3]:
        ax.legend()

    ax3.set_xlabel("Time (s)")
    ax2.set_ylabel("Displacement")

    plt.savefig(filename.strip(".xlsx")+".png")
    # plt.show()
    plt.close()

    ht_ave = np.average(df["Height (mm)"][t0:t0+num_pts])

    return pars, linpars, res_sum, ht_ave


def z_fit(t, A, f, omega0, Qm, delta, fall):
    t2 = t + delta      # add a phase shift
    omega = 2 * np.pi * f  # driving force angular frequency
    return z(t2, A, omega, omega0, Qm) + fall*t2


def fit_response(folder, filename, A0=1., f=1., omega0=1., Qm=10.):
    # plot driving, relative and response displacements
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, )

    num_pts = 1500
    t = np.linspace(0, num_pts*0.02, num=num_pts)

    path = os.path.join(folder, filename)
    df = pd.read_excel(path, sheet_name="RFC data")
    t0 = 2300
    z_real = df["LVDT (mm)"][t0:t0+num_pts] - np.average(df["LVDT (mm)"][t0:t0+num_pts])
    # plt.plot(t, z_real)
    # plt.show()

    # Fit using z above
    try:
        z_pars, z_cov = curve_fit(f=z_fit, xdata=t, ydata=z_real, p0=[A0, f, omega0, Qm, 0.1, -3e-03], bounds=(-np.inf, np.inf))

        # print("A0: {}, f: {}, omega0: {}, Qm: {}, delta: {}, fall: {}".format(*z_pars))
        # z_theor = z(t, A0, omega, omega0, Qm)

        ax1.plot(t, z_real, label="Expt")
        # ax1.plot(t, sin_cos_fit(t, *pars), label='Fit')

        ax2.plot(t, z_real, label="Expt")
        ax2.plot(t, z_fit(t, *z_pars), label="Model")

        # ax3.plot(t, z_real - sin_fit(t, *pars), label='Difference sin')
        ax3.plot(t, z_real - z_fit(t, *z_pars), label="Difference model")

        for ax in [ax1, ax2, ax3]:
            ax.legend()

        ax3.set_xlabel("Time (s)")
        ax2.set_ylabel("Displacement")

        plt.show()

        return z_pars

    except RuntimeError as e:
        # print(e)
        plt.plot(t, z_real)
        plt.show()
        return e, [], [], [], [], []


def do_fft(folder, fname):
    f_list = fname.split("_")
    # fig, (ax1, ax3) = plt.subplots(2, 1, sharex=False, )
    vol = f_list[1]
    flo = f_list[2]

    # Get data
    path = os.path.join(folder, fname)
    df = pd.read_excel(path, sheet_name="RFC data")

    z_real = df["LVDT (V)"][2100:4100] - np.average(df["LVDT (V)"][2100:4100])
    t = np.linspace(0, 2000 * 0.02, num=2000)

    # ax1.plot(t, z_real)
    # ax1.set_title('Displacement due to SP motion: {} mL at {} mL/min'.format(vol, flo))
    # ax1.set_xlabel('Time (s)')
    # ax1.set_ylabel('Position (mm)')

    # Frequency domain representation
    amplitude = z_real
    samplingFrequency = 1/0.02

    fourierTransform_raw = np.fft.fft(amplitude)

    fourierTransform_range = fourierTransform_raw[range(int(len(amplitude) / 2))]    # Exclude sampling frequency
    fourierTransform = fourierTransform_range / len(amplitude)  # Normalize amplitude

    tpCount = len(amplitude)
    values = np.arange(int(tpCount / 2))
    timePeriod = tpCount / samplingFrequency
    frequencies = values / timePeriod

    # Frequency domain representation

    # ax3.set_title('Fourier transform depicting the frequency components')
    # ax3.semilogy(frequencies, abs(fourierTransform))
    # ax3.set_xlabel('Frequency (Hz)')
    # ax3.set_ylabel('Relative Amplitude')
    #
    # plt.tight_layout()
    save_name = os.path.join(folder, "FFT_{}_{}_{}.png".format(vol, flo, f_list[3]))
    # plt.savefig(save_name)
    # plt.show()

    return [vol, flo, frequencies, fourierTransform_range]

def plot_all_FFTs(folder):

    ffts = []
    for f in get_all_fnames(folder, "Tri", endpattern="_LVDT.xlsx")[10:20]:
        print(f)
        ffts.append(do_fft(folder, f))

    n_col = len(ffts)
    color = plt.cm.rainbow(np.linspace(0, 1, n_col))
    for i, dataset in enumerate(ffts):
        plt.semilogy(dataset[2], abs(dataset[3]), label="{} @ {}".format(dataset[0], dataset[1]), c=color[i])
        # print(np.argmax(max(abs(dataset[3]))))

    plt.title('Fourier transform depicting the frequency components')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    """Parameters"""
    t_final = 60
    num_pts = 100
    t = np.linspace(0, t_final, int(num_pts * t_final + 1))

    f0 = 1.035  # natural frequency, which is around 1 Hz for current setup
    omega0 = 2 * np.pi * f0

    Q = 7  # Quality factor - higher for low damping and longer oscillations
    m = 2  # Mass of floating elements, kg
    Qm = Q * m
    # omega1 = np.sqrt(omega0**2 - 0.5 * gamma**2)

    f = 0.933  # driving force frequency
    omega = 2 * np.pi * f  # driving force angular frequency
    A0 = 0.1

    psi = 4

    folder_0p05 = r"G:\Shared drives\MSL - Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20201109 TriangleWaves 0.05mL"

    plot_all_FFTs(folder_0p05)
    #
    # filename = "Tri_0.05_18_1604882252.6072_LVDT.xlsx"
    # vol = float(filename.split("_")[1])
    # flo = float(filename.split("_")[2])
    # f = 1 / (2 * vol / flo * 60 + 0.03)
    #
    # pars, linpars = fit_response_single_sine(folder_0p05, filename, A0=A0, f=f, )
    # # returns [A, f, phi], [fall, offset]
    # print([filename, pars[0], pars[1], pars[2], linpars[0], linpars[1]])


    # fout = Workbook()
    # sh = fout.active
    #
    # files = get_all_fnames(folder_0p05, "Tri_", endpattern="_LVDT.xlsx")
    #
    # # files = ["Tri_0.05_18_1604882252.6072_LVDT.xlsx", "Tri_0.05_15_1604882091.3002_LVDT.xlsx"]
    #
    # for filename in files:
    #     print(filename)
    #
    #     vol = float(filename.split("_")[1])
    #     flo = float(filename.split("_")[2])
    #     f = 1/(2*vol/flo*60+0.025)
    #
    #     pars, linpars, res_sum, ht_ave = fit_response_single_sine(folder_0p05, filename, A0=A0, f=f, )
    #     # returns [A, f, phi], [fall, offset]
    #     sh.append([filename, pars[0], pars[1], pars[2], linpars[0], linpars[1], res_sum, ht_ave])
    #     # print("{}, A0: {}, f: {}, omega0: {}, Qm: {}, delta: {}, fall: {}".format(filename, *z_pars))
    #
    # fout.save("SineOsc.xlsx")



