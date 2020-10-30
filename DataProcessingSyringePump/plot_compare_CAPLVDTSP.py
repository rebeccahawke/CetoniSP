import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import scipy.optimize as opt


def linear(x, m, c):
     return m * x + c

msl_orange = '#f15a29'


h = 3 #6.26/2.54
w = 4 #7.98/2.54

ramps = True
measured_v_nominal = False
measured_v_measured = False
vel_v_flow = True

path = r'C:\Users\r.hawke\Desktop\4OctFiles\Working folder for APMF paper'

data_SP = pd.read_csv(path + '\\' + 'summary_data_SP.csv', index_col='ID').sort_index()
data_LVDT = pd.read_csv(path + '\\' + 'summary_data_LVDT.csv', index_col='ID').sort_index()
data_CAP = pd.read_csv(path + '\\' + 'summary_data_CAP.csv', index_col='ID').sort_index()

data_inf = pd.read_csv(path + '\\' + '20191002_step_25sccm_i_25.0_1569978651_alldata.csv')
data_wdr = pd.read_csv(path + '\\' + '20191002_step_25sccm_w_25.0_1569978536_alldata.csv')

# print(data_SP.index == data_CAP.index)

piston_area = 196.12 # mm2
syringe_dia = 23.03 # mm
syringe_area = np.pi * (syringe_dia/2)**2

mult = 0.001*60 # multiplier mm/2 to ccm

syringe_flo = data_SP["Travel m"]*-1*mult*syringe_area
syr_error = data_SP['u(Travel m)']

nom_flo = data_SP['Flowrate (sccm)']
for id in data_SP.index:
    nom_flo.loc[id] = -1*data_SP.loc[id, 'Flowrate (sccm)']*np.sign(data_SP.loc[id, 'Travel m'])

piston_travel_LVDT = data_LVDT["Travel m"]
piston_flo_LVDT = piston_travel_LVDT * mult * piston_area

piston_travel_CAP = data_CAP["Travel m"]
piston_flo_CAP = piston_travel_CAP * mult * piston_area
piston_error = data_CAP['u(Travel m)']
timeint = 0.05

#########################################################################################
### PLOT RAMPS  ###

if ramps:
    # curve fitting of rates - infusion
    time_inf = data_inf['Time (s)']
    height_inf = data_inf['Height_rfc']

    t1_inf = 29.7
    t2_inf = 30.7
    t_sub = 20
    i1_inf = int(t1_inf / timeint)
    p2_inf = int(t2_inf / timeint)
    p0_inf = i1_inf - int(10 / timeint)
    p3_inf = p2_inf + int(20 / timeint)

    linParamsT_inf, pcovT_inf = opt.curve_fit(linear, time_inf[i1_inf:p2_inf], height_inf[i1_inf:p2_inf])  # piston travel rate

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

    # Plot data and fits
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[w, h*2])
    axins1 = inset_axes(ax1, width="100%", height="100%",
                    bbox_to_anchor=(.65, .3, 1/3, 1/4),
                    bbox_transform=ax1.transAxes)
    axins2 = inset_axes(ax2, width="100%", height="100%",
                    bbox_to_anchor=(.65, .4, 1/3, 1/4),
                    bbox_transform=ax2.transAxes)

    ax1.plot(time_inf[p0_inf:p3_inf] - t_sub, height_inf[p0_inf:p3_inf],
             label="Data", marker='.', color=msl_orange, linestyle='None')
    ax1.plot(time_inf[i1_inf:p2_inf] - t_sub, linear(time_inf[i1_inf:p2_inf], *linParamsT_inf), color='k',
             label="Linear fit with gradient = " + str(round(linParamsT_inf[0], 3)) + "\n(u = " + str(
                 round(pcovT_inf[0][0], 3)) + ") mm/s")

    axins1.plot(time_inf[i1_inf:p2_inf] - 20, height_inf[i1_inf:p2_inf] - linear(time_inf[i1_inf:p2_inf], *linParamsT_inf),
                label="Data", marker='.', color=msl_orange, linestyle='None')

    ax2.plot(time_wdr[p0_wdr:p3_wdr] - t_sub, height_wdr[p0_wdr:p3_wdr],
             label="Data", marker='.', color=msl_orange, linestyle='None')
    ax2.plot(time_wdr[i1_wdr:p2_wdr] - t_sub, linear(time_wdr[i1_wdr:p2_wdr], *linParamsT_wdr), color='k',
             label="Linear fit with gradient = " + str(round(linParamsT_wdr[0], 3)) + "\n(u = " + str(
                 round(pcovT_wdr[0][0], 3)) + ") mm/s")

    axins2.plot(time_wdr[i1_wdr:p2_wdr] - 20, height_wdr[i1_wdr:p2_wdr] - linear(time_wdr[i1_wdr:p2_wdr], *linParamsT_wdr),
                label="Data", marker='.', color=msl_orange, linestyle='None')

    # Show the graph
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True) #(loc='upper center')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Piston position (mm)')
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
    ax2.set_ylim(7.0,10.7)
    ax2.text(-0.15, 1.1, '(b)', transform=ax2.transAxes, size=12,)
    ax2.axvspan(t1_wdr - t_sub, t2_wdr - t_sub, alpha=0.5, color='silver')

    axins2.set_xlabel('Time (s)')
    axins2.set_ylabel('Res. (mm)')
    axins2.set_xlim(t1_wdr - t_sub, t2_wdr - t_sub)
    axins2.set_ylim(-0.27, 0.27)
    axins2.axvspan(t1_wdr - t_sub, t2_wdr - t_sub, alpha=0.5, color='silver')
    # axins2.axhspan(-0.05, 0.05, alpha=0.5, color='cornflowerblue')

    fig.subplots_adjust(hspace=0.3)

    # fig.show()

    fig.savefig(path + '\\' + '20191002_step_25sccm_inf_wdr_1200_orange.png', dpi=1200)
    # fig.savefig(path + '\\' + '20191002_step_25sccm_inf_wdr.pdf')
    fig.savefig(path + '\\' + '20191002_step_25sccm_inf_wdr_orange.svg')


#########################################################################################
### PLOT MEASURED VS NOMINAL  ###

if measured_v_nominal:
    # curve fitting of leak rates
    # linParamsCAP, pcovCAP = opt.curve_fit(linear, nom_flo[:], piston_flo_CAP)
    # linParamsLVDT, pcovLVDT = opt.curve_fit(linear, nom_flo[:], piston_flo_LVDT)

    # Plot data and fits
    fig = plt.figure(figsize=(w, h))
    # plt.plot(nom_flo, piston_flo_LVDT - nom_flo, label='LVDT', marker='o', color='k', fillstyle='none', linestyle='None')
    # plt.plot(nom_flo, piston_flo_CAP - nom_flo, label='Capacitor', marker='*', color='b', linestyle='none')
    # plt.scatter(nom_flo, syringe_flo - nom_flo)#, label='Syringe')#, marker='.', color='g', linestyle='none')
    plt.errorbar(nom_flo, syringe_flo - nom_flo, yerr=syr_error, fmt='o', markersize=4, color=msl_orange, )
    # plt.plot(syringe_flo, linear(syringe_flo, *linParamsLVDT), label="Fit LVDT")
    # plt.plot(syringe_flo, linear(syringe_flo, *linParamsCAP), linestyle=':', color='k',
    #          label="Linear fit with \ngradient = " + str(round(linParamsCAP[0], 4)) + "\n(u = " + str(
    #              round(pcovCAP[0][0], 8)) + ")")

    # Show the graph
    # plt.legend()
    # plt.title("Data from CAP")
    # plt.suptitle("m, c: "+str(linParamsCAP))

    plt.xlabel('Nominal flow rate (ccm)')
    plt.ylabel('Measured - nominal flow rate (ccm)')
    plt.tight_layout(.5)

    # plt.show()

    fig.savefig('20191002_step_summary_syr-nom_orange.png', dpi=1200)
    # fig.savefig(path + '\\' + '20191002_step_summary_syr-nom.pdf')
    fig.savefig(path + '\\' + '20191002_step_summary_syr-nom_orange.svg')


#########################################################################################
### PLOT MEASURED VS MEASURED  ###

if measured_v_measured:
    # curve fitting of leak rates
    linParamsCAP, pcovCAP = opt.curve_fit(linear, syringe_flo[:], piston_flo_CAP)
    # linParamsLVDT, pcovLVDT = opt.curve_fit(linear, syringe_flo[:], piston_flo_LVDT)

    # Plot data and fits
    fig = plt.figure(figsize=(w,h))
    # plt.scatter(syringe_flo, piston_flo_LVDT, label='LVDT')
    # plt.errorbar(syringe_flo, piston_flo_CAP, yerr=piston_error, fmt='o', markersize=4, label='Data',)
    plt.scatter(syringe_flo, piston_flo_CAP, label='Data', color=msl_orange, )
    # plt.plot(syringe_flo, linear(syringe_flo, *linParamsLVDT), label="Fit LVDT")
    plt.plot(syringe_flo, linear(syringe_flo, *linParamsCAP), color='k',
             label="Linear fit with \ngradient = "+str(round(linParamsCAP[0],4))+"\n(u = "+ str(round(pcovCAP[0][0],8)) +")")

    # Show the graph
    plt.legend()
    # plt.title("Data from CAP")
    # plt.suptitle("m, c: "+str(linParamsCAP))

    plt.xlabel('Measured flow rate at syringe pump (ccm)')
    plt.ylabel('Measured flow rate at piston (ccm)')
    plt.tight_layout(.5)

    # plt.show()

    fig.savefig(path + '\\' + '20191002_step_summary_CAP_orange.png', dpi=1200)
    # fig.savefig(path + '\\' + '20191002_step_summary_CAP.pdf')
    fig.savefig(path + '\\' + '20191002_step_summary_CAP_orange.svg')


#########################################################################################
### PLOT PISTON VELOCITY VS SYRINGE FLOW  ###

if vel_v_flow:
    # curve fitting of leak rates
    linParamsCAP, pcovCAP = opt.curve_fit(linear, syringe_flo[:], piston_travel_CAP)
    # linParamsLVDT, pcovLVDT = opt.curve_fit(linear, syringe_flo[:], piston_flo_LVDT)

    # Plot data and fits
    fig = plt.figure(figsize=(w,h))
    # plt.scatter(syringe_flo, piston_flo_LVDT, label='LVDT')
    # plt.errorbar(syringe_flo, piston_flo_CAP, yerr=piston_error, fmt='o', markersize=4, label='Data',)
    plt.scatter(syringe_flo, piston_travel_CAP, label='Data', color=msl_orange, )
    # plt.plot(syringe_flo, linear(syringe_flo, *linParamsLVDT), label="Fit LVDT")
    plt.plot(syringe_flo, linear(syringe_flo, *linParamsCAP), color='k',
             label="Linear fit with \ngradient = "+str(round(linParamsCAP[0],4))+"\n(u = "+ str(round(pcovCAP[0][0],8)) +")")

    # Show the graph
    plt.legend()
    # plt.title("Data from CAP")
    # plt.suptitle("m, c: "+str(linParamsCAP))

    plt.xlabel('Measured flow rate at syringe pump (ccm)')
    plt.ylabel('Measured velocity of piston (mm $s^{-1}$)')
    plt.xlim(-30, 35)
    plt.tight_layout(.5)

    # plt.show()

    fig.savefig(path + '\\' + '20191002_step_vel-v-flo_CAP_orange.png', dpi=1200)
    # fig.savefig(path + '\\' + '20191002_step_summary_CAP.pdf')
    fig.savefig(path + '\\' + '20191002_step_vel-v-flo_CAP_orange.svg')