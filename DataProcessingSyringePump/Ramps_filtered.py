import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import scipy.optimize as opt

def linear(x, m, c):
    return m * x + c

def sinusoid(t, A, f, phi):
    return A * np.cos(2 * np.pi * f * t - phi)

msl_orange = '#f15a29'

h = 3  # 6.26/2.54
w = 4  # 7.98/2.54

ramps = True

path = r'C:\Users\r.hawke\Desktop\4OctFiles\Working folder for APMF paper'

data_inf = pd.read_csv(path + '\\' + '20191002_step_25sccm_i_25.0_1569978651_alldata.csv')
data_wdr = pd.read_csv(path + '\\' + '20191002_step_25sccm_w_25.0_1569978536_alldata.csv')


timeint = 0.05

#########################################################################################
### PLOT RAMPS  ###

if ramps:
    # curve fitting of rates - infusion
    time_inf = data_inf['Time (s)']
    height_inf = data_inf['Height_rfc']

    t1_inf = 29.7
    t2_inf = 30.7
    t_sub = 19.7
    t_sub2 = 20.6
    i1_inf = int(t1_inf / timeint)
    p2_inf = int(t2_inf / timeint)
    p0_inf = i1_inf - int(10 / timeint)
    p3_inf = p2_inf + int(20 / timeint)

    linParamsT_inf, pcovT_inf = opt.curve_fit(linear, time_inf[i1_inf:p2_inf], height_inf[i1_inf:p2_inf])  # piston travel rate

    # filtering of noise - infusion
    A1 = -0.0334413
    f1 = 1.05485
    phi1 = 8.90099

    A3 = 0.0308766
    f3 = 1.0135
    phi3 = -2.33181

    V_fit1 = sinusoid(time_inf[:p2_inf], A1, f1, phi1)
    V_fit3 = sinusoid(time_inf[p2_inf:], A3, f3, phi3)

    # curve fitting of rates - withdrawal
    time_wdr = data_wdr['Time (s)']
    height_wdr = data_wdr['Height_rfc']

    t1_wdr = 30.6
    t2_wdr = 31.95
    i1_wdr = int(t1_wdr / timeint)
    p2_wdr = int(t2_wdr / timeint)
    p0_wdr = i1_wdr - int(10 / timeint)
    p3_wdr = p2_wdr + int(20 / timeint)

    linParamsT_wdr, pcovT_wdr = opt.curve_fit(linear, time_wdr[i1_wdr:p2_wdr], height_wdr[i1_wdr:p2_wdr])  # piston travel rate

    # filtering of noise - withdrawal
    A2 = 0.0297139
    f2 = 1.20941
    phi2 = 0.000846732

    A4 = 0.0332837
    f4 = 1.16115
    phi4 = 3.30832

    V_fit2 = sinusoid(time_inf, A2, f2, phi2)
    V_fit4 = sinusoid(time_inf, A4, f4, phi4)


    # Plot data and fits
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[w, h*2])
    axins1 = inset_axes(ax1, width="100%", height="100%",
                    bbox_to_anchor=(.62, .3, 1/3, 1/4),
                    bbox_transform=ax1.transAxes)
    axins2 = inset_axes(ax2, width="100%", height="100%",
                    bbox_to_anchor=(.66, .4, 1/3, 1/4),
                    bbox_transform=ax2.transAxes)

    ax1.plot(time_inf[:p2_inf] - t_sub, height_inf[:p2_inf] - V_fit1,
             label="Data", marker='.', color=msl_orange, linestyle='None')
    ax1.plot(time_inf[p2_inf:] - t_sub, height_inf[p2_inf:] - V_fit3,
             marker='.', color=msl_orange, linestyle='None')
    # ax1.plot(time_inf[p0_inf:p3_inf] - t_sub, height_inf[p0_inf:p3_inf],
    #          label="Data", marker='.', color=msl_orange, linestyle='None')
    ax1.plot(time_inf[i1_inf:p2_inf] - t_sub, linear(time_inf[i1_inf:p2_inf], *linParamsT_inf), color='k',
             label="Linear fit with gradient = " + str(round(linParamsT_inf[0], 3)) + "\n(u = " + str(
                 round(pcovT_inf[0][0], 3)) + ") mm/s")

    axins1.plot(time_inf[i1_inf:p2_inf] - t_sub,
                height_inf[i1_inf:p2_inf] - linear(time_inf[i1_inf:p2_inf] - V_fit1[i1_inf:p2_inf], *linParamsT_inf),
                label="Data", marker='.', color=msl_orange, linestyle='None')

    ax2.plot(time_wdr[:p2_wdr] - t_sub2, height_wdr[:p2_wdr] - V_fit2[:p2_wdr],
             label="Data", marker='.', color=msl_orange, linestyle='None')
    ax2.plot(time_wdr[p2_wdr:p3_wdr] - t_sub2, height_wdr[p2_wdr:p3_wdr] - V_fit4[p2_wdr:p3_wdr],
             marker='.', color=msl_orange, linestyle='None')
    ax2.plot(time_wdr[i1_wdr:p2_wdr] - t_sub2, linear(time_wdr[i1_wdr:p2_wdr], *linParamsT_wdr), color='k',
             label="Linear fit with gradient = " + str(round(linParamsT_wdr[0], 3)) + "\n(u = " + str(
                 round(pcovT_wdr[0][0], 3)) + ") mm/s")

    axins2.plot(time_wdr[i1_wdr:p2_wdr] - t_sub2,
                height_wdr[i1_wdr:p2_wdr] - linear(time_wdr[i1_wdr:p2_wdr] - V_fit2[i1_wdr:p2_wdr], *linParamsT_wdr),
                label="Data", marker='.', color=msl_orange, linestyle='None')

    # Show the graph
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True) #(loc='upper center')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Piston position (mm)')
    ax1.set_xlim(0, 30)
    ax1.set_ylim(6.8,10.5)
    ax1.text(-0.15, 1.1, '(a)', transform=ax1.transAxes, size=12,)
    ax1.axvspan(t1_inf - t_sub, t2_inf - t_sub, alpha=0.5, color='silver')

    axins1.set_xlabel('Time (s)')
    axins1.set_ylabel('Res. (mm)')
    axins1.set_xlim(t1_inf - t_sub, t2_inf - t_sub)
    axins1.set_ylim(-.23, 0.23)
    axins1.axvspan(t1_inf - t_sub, t2_inf - t_sub, alpha=0.5, color='silver')
    # axins1.axhspan(-0.05, 0.05, alpha=0.5, color='cornflowerblue')

    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True) #(loc=9)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Piston position (mm)')
    ax2.set_xlim(0, 30)
    ax2.set_ylim(7.0,10.7)
    ax2.text(-0.15, 1.1, '(b)', transform=ax2.transAxes, size=12,)
    ax2.axvspan(t1_wdr - t_sub2, t2_wdr - t_sub2, alpha=0.5, color='silver')

    axins2.set_xlabel('Time (s)')
    axins2.set_ylabel('Res. (mm)')
    axins2.set_xlim(t1_wdr - t_sub2, t2_wdr - t_sub2)
    axins2.set_ylim(-0.27, 0.27)
    axins2.axvspan(t1_wdr - t_sub2, t2_wdr - t_sub2, alpha=0.5, color='silver')
    # axins2.axhspan(-0.05, 0.05, alpha=0.5, color='cornflowerblue')

    fig.subplots_adjust(hspace=0.3)

    # fig.show()

    fig.savefig(path + '\\' + '20191002_step_25sccm_inf_wdr_1200_orange_filtered.png', dpi=1200)
    # fig.savefig(path + '\\' + '20191002_step_25sccm_inf_wdr.pdf')
    fig.savefig(path + '\\' + '20191002_step_25sccm_inf_wdr_orange_filtered.svg')