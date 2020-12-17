import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import scipy.optimize as opt


def linear(x, m, c):
    return m * x + c


def dampedHO(t, A, omega0, Q, psi, slope, offset):
    gamma = omega0 / Q        # here gamma = Gamma/m; this relation is true for a weakly damped oscillator
    # omega = np.sqrt(omega0 ** 2 - (gamma/2) ** 2)
    omega = omega0 * np.sqrt(1 - 1/(4 * Q**2))
    return A * np.exp(-gamma / 2 * t) * np.sin(omega * (t - psi)) + slope * t + offset


msl_orange = '#f15a29'

h = 3  # 6.26/2.54
w = 4  # 7.98/2.54

path = r'G:\Shared drives\MSL - Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20201209 Steps 0.75mL'

#
# xls = pd.ExcelFile(f1)
# starts = pd.read_excel(xls, 'Sheet1')

data_inf = pd.read_excel(os.path.join(path, 'Step_0.75_25_1607473103.6122003_all.xlsx'), 'RFC data')
data_wdr = pd.read_excel(os.path.join(path, 'Step_-0.75_25_1607473015.912431_all.xlsx'), 'RFC data')

ramps = True
timeint = 0.01

#########################################################################################
### PLOT RAMPS  ###

if ramps:
    # curve fitting of rates - infusion
    time_inf = data_inf['Time (s)']
    # time_delta = datetime.strptime(time_inf[2], '%Y-%m-%d %H:%M:%S.%f')-datetime.strptime(time_inf[1], '%Y-%m-%d %H:%M:%S.%f')
    time_int = time_inf[1] - time_inf[0]
    height_inf = data_inf['LVDT (mm)']

    t1_inf = 20.18  # start time for ramp
    t2_inf = 21.9  # end time for ramp
    t_sub = t1_inf
    i1_inf = int(t1_inf / timeint)
    p2_inf = int(t2_inf / timeint)
    p0_inf = i1_inf - int(10 / timeint)
    p3_inf = p2_inf + int(20 / timeint)

    # piston fall rate
    linParamsFall_inf, pcovFall_inf = opt.curve_fit(linear, time_inf[1:i1_inf], height_inf[1:i1_inf])

    # piston travel rate
    linParamsT_inf, pcovT_inf = opt.curve_fit(linear, time_inf[i1_inf:p2_inf], height_inf[i1_inf:p2_inf])

    # piston oscillation post ramp
    hoParams_inf, pcovHO_inf = opt.curve_fit(dampedHO, time_inf[p2_inf:], height_inf[p2_inf:],
                                             p0=[1, 1, 7, 0, -0.0001, np.average(height_inf[p2_inf:])],
                                             bounds=(-100, 100))

    # curve fitting of rates - withdrawal
    time_wdr = data_wdr['Time (s)']
    height_wdr = data_wdr['LVDT (mm)']

    t1_wdr = t1_inf
    t2_wdr = t2_inf
    t_sub2 = t1_inf
    i1_wdr = int(t1_wdr / timeint)
    p2_wdr = int(t2_wdr / timeint)
    p0_wdr = i1_wdr - int(10 / timeint)
    p3_wdr = p2_wdr + int(20 / timeint)

    # piston fall rate
    linParamsFall_wdr, pcovFall_wdr = opt.curve_fit(linear, time_wdr[1:i1_wdr], height_wdr[1:i1_wdr])

    # piston travel rate
    linParamsT_wdr, pcovT_wdr = opt.curve_fit(linear, time_wdr[i1_wdr:p2_wdr], height_wdr[i1_wdr:p2_wdr])

    # piston oscillation post ramp
    hoParams_wdr, pcovHO_wdr = opt.curve_fit(dampedHO, time_wdr[p2_wdr:], height_wdr[p2_wdr:], p0=[1, 1, 7, 0, -0.0001, np.average(height_wdr[p2_wdr:])], bounds=(-100, 100))

    # Plot data and fits
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[w, h*2])
    axins1 = inset_axes(ax1, width="100%", height="100%",
                    bbox_to_anchor=(.66, .3, 1/3, 1/4),
                    bbox_transform=ax1.transAxes)
    axins2 = inset_axes(ax2, width="100%", height="100%",
                    bbox_to_anchor=(.66, .4, 1/3, 1/4),
                    bbox_transform=ax2.transAxes)

    ax1.plot(time_inf[:p2_inf] - t_sub, height_inf[:p2_inf],
             label="Data", marker='.', color=msl_orange, linestyle='None')
    ax1.plot(time_inf[p2_inf:] - t_sub, height_inf[p2_inf:],
             marker='.', color=msl_orange, linestyle='None')
    ax1.plot(time_inf[1:i1_inf] - t_sub, linear(time_inf[1:i1_inf], *linParamsFall_inf), color='k', linestyle='--',
             label="Linear fit with gradient = " + str(round(linParamsFall_inf[0], 4)) + "\n(u = " + str(
                 round(pcovFall_inf[0][0], 4)) + ") mm/s")
    ax1.plot(time_inf[i1_inf:p2_inf] - t_sub, linear(time_inf[i1_inf:p2_inf], *linParamsT_inf), color='k',
             label="Linear fit with gradient = " + str(round(linParamsT_inf[0], 4)) + "\n(u = " + str(
                 round(pcovT_inf[0][0], 4)) + ") mm/s")
    ax1.plot(time_inf[p2_inf:] - t_sub, dampedHO(time_inf[p2_inf:], *hoParams_inf), color='k',
             linestyle='--', label="Damped HO fit")

    axins1.plot(time_inf[i1_inf:p2_inf] - t_sub,
                height_inf[i1_inf:p2_inf] - linear(time_inf[i1_inf:p2_inf], *linParamsT_inf),
                label="Data", marker='.', color=msl_orange, linestyle='None')

    ax2.plot(time_wdr[:p2_wdr] - t_sub2, height_wdr[:p2_wdr],
             label="Data", marker='.', color=msl_orange, linestyle='None')
    ax2.plot(time_wdr[p2_wdr:p3_wdr] - t_sub2, height_wdr[p2_wdr:p3_wdr],
             marker='.', color=msl_orange, linestyle='None')
    ax2.plot(time_wdr[1:i1_wdr] - t_sub2, linear(time_wdr[1:i1_wdr], *linParamsFall_wdr), color='k', linestyle='--',
             label="Linear fit with gradient = " + str(round(linParamsFall_wdr[0], 4)) + "\n(u = " + str(
                 round(pcovFall_wdr[0][0], 4)) + ") mm/s")
    ax2.plot(time_wdr[i1_wdr:p2_wdr] - t_sub2, linear(time_wdr[i1_wdr:p2_wdr], *linParamsT_wdr), color='k',
             label="Linear fit with gradient = " + str(round(linParamsT_wdr[0], 4)) + "\n(u = " + str(
                 round(pcovT_wdr[0][0], 4)) + ") mm/s")
    ax2.plot(time_wdr[p2_wdr:] - t_sub2, dampedHO(time_wdr[p2_wdr:], *hoParams_wdr), color='k',
             linestyle='--', label="Damped HO fit")

    axins2.plot(time_wdr[i1_wdr:p2_wdr] - t_sub2,
                height_wdr[i1_wdr:p2_wdr] - linear(time_wdr[i1_wdr:p2_wdr], *linParamsT_wdr),
                label="Data", marker='.', color=msl_orange, linestyle='None')

    # Show the graph
    ax1.legend() #bbox_to_anchor=(0.5, 1.05),
        #  ncol=3, fancybox=True, shadow=True) #loc='upper center',

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (mm)')  # Piston position
    ax1.set_xlim(-10, 20)
    ax1.set_ylim(-0.5, 5.9)
    ax1.text(-0.15, 1.1, '(a)', transform=ax1.transAxes, size=12,)
    ax1.axvspan(t1_inf - t_sub, t2_inf - t_sub, alpha=0.5, color='silver')

    axins1.set_xlabel('Time (s)')
    axins1.set_ylabel('Res. (mm)')
    axins1.set_xlim(t1_inf - t_sub, t2_inf - t_sub)
    axins1.set_ylim(-.4, 0.4)
    axins1.axvspan(t1_inf - t_sub, t2_inf - t_sub, alpha=0.5, color='silver')
    # axins1.axhspan(-0.05, 0.05, alpha=0.5, color='cornflowerblue')

    ax2.legend() #loc='upper center', bbox_to_anchor=(0.5, 1.05),
       #   ncol=3, fancybox=True, shadow=True) #

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (mm)   ')
    ax2.set_xlim(-10, 20)
    ax2.set_ylim(-0.5, 5.9)
    ax2.text(-0.15, 1.1, '(b)', transform=ax2.transAxes, size=12,)
    ax2.axvspan(t1_wdr - t_sub2, t2_wdr - t_sub2, alpha=0.5, color='silver')

    axins2.set_xlabel('Time (s)')
    axins2.set_ylabel('Res. (mm)')
    axins2.set_xlim(t1_wdr - t_sub2, t2_wdr - t_sub2)
    axins2.set_ylim(-0.4, 0.4)
    axins2.axvspan(t1_wdr - t_sub2, t2_wdr - t_sub2, alpha=0.5, color='silver')
    # axins2.axhspan(-0.05, 0.05, alpha=0.5, color='cornflowerblue')

    fig.subplots_adjust(hspace=0.3)

    new_path = os.path.join(path, 'Step_0.75_25ccm_moreFitting')
    fig.savefig(new_path + '.png', dpi=1200)
    fig.savefig(new_path + '.svg')

    plt.show()
