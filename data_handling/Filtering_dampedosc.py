import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from openpyxl import Workbook, load_workbook

from data_handling.process_files import get_all_fnames


def linear(x, m, c):
     return m * x + c

msl_orange = '#f15a29'


def dampedHO(t, A, omega0, Q, psi, slope, offset):
    gamma = omega0 / Q        # here gamma = Gamma/m; this relation is true for a weakly damped oscillator
    # omega = np.sqrt(omega0 ** 2 - (gamma/2) ** 2)
    omega = omega0 * np.sqrt(1 - 1/(4 * Q**2))
    return A * np.exp(-gamma / 2 * t) * np.sin(omega * (t - psi)) + slope * t + offset


def get_all_LVDTxlsx_fnames(mydir):
    files = []

    for file in os.listdir(mydir):
        if file.endswith("_LVDT.xlsx"):
            files.append(file)

    return files


def fit_damping(folder, filename, t_start=None, end=None):
    xls = pd.ExcelFile(os.path.join(folder, filename))
    # df1 = pd.read_excel(xls, 'SP data')
    df2 = pd.read_excel(xls, 'RFC data')

    # t_params = pd.read_csv(path+'\\'+'t_params.csv', index_col='ID')

    file_list = filename.split("_")
    set = file_list[1]
    flowrate = file_list[2]
    run = file_list[3]
    ID = set + flowrate + run

    x = df2["Timestamp"][1:]
    x_t = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S.%f') for i in x]
    x_d = [(i - x_t[0]).total_seconds() for i in x_t]
    height = df2["LVDT (mm)"][1:]
    # t_start = df2["LVDT (V)"][0]
    print(t_start)

    time_delta = datetime.strptime(x[2], '%Y-%m-%d %H:%M:%S.%f')-datetime.strptime(x[1], '%Y-%m-%d %H:%M:%S.%f')
    timestep = time_delta.total_seconds()
    print(timestep)

    time = np.linspace(0, x_d[-1], len(x_d))
    print(time)

    if t_start is None:
        plt.plot(time, height,
                 label="Data", marker='.', color=msl_orange, linestyle='None')
        plt.show()
        t_start = float(input("t start"))

    a = t_start

    if end is not None:
        b = end
    else:
        b = a + 5

    pointA = int(a / timestep)
    pointB = int(b / timestep)

    plt.figure(figsize=(4,2))

    plt.plot(time, height,
                label="Data", marker='.', color=msl_orange, linestyle='None')

    plt.xlabel('Time (s)')
    plt.ylabel('Piston position (mm)')
    plt.tight_layout()

    if t_start is not None:
        plt.xlim(a, b)

        time_synth = np.linspace(0, t_start+10, int(100*(t_start+10)+1))

        t = time[pointA:pointB] - time[pointA]
        h = height[pointA:pointB]

        try:
            optParams, pcov = curve_fit(dampedHO, t, h, p0=[1, 1, 7, 0, -0.0001, np.average(h)], bounds=(-100, 100))
            print(optParams)
            print('Amplitude =', optParams[0])
            print('Omega0 =', optParams[1])
            print('T0 = {}, f0 = {}'.format(2*np.pi/optParams[1], optParams[1]/(2*np.pi)))
            print('Q =', optParams[2])
            print('Phase =', optParams[3])
            print('Slope =', optParams[4])
            print('Offset =', optParams[5])

            # print("{} {} {}".format(t_start, optParams[1]/(2*np.pi), optParams[2]))

            fit_osc = dampedHO(time_synth, optParams[0], optParams[1], optParams[2], optParams[3], slope=optParams[4], offset=optParams[5] )
            fit = plt.plot(time_synth+t_start, fit_osc,
                        label='Model', color='k', alpha=0.5)
            res = [y_ - x_ for y_, x_ in zip(height[pointA:pointB], fit_osc)]
            stdres = np.std(res, ddof=1)

            plt.plot(time_synth[pointA:pointB], res)

            plt.legend()

            # plt.savefig(savepath + '\\' + '20191002_set1_r25sccm_infuse1_osc_model.svg')
            # plt.savefig(savepath + '\\' + '20191002_set1_r25sccm_infuse1_osc_model.png', dpi=1200)
            plt.tight_layout()
            plt.show()
            plt.close()
            return t_start, optParams[1]/(2*np.pi), optParams[2], optParams[4], optParams[5], stdres

        except RuntimeError as e:
            print(e)
            return t_start, 0, 0, 0, 0


folder_0p5 = r"C:\Users\r.hawke\PycharmProjects\CetoniSP\data_files\2020-10-23 TriangleWaves 0.5mL"
folder_1Hz = r"C:\Users\r.hawke\PycharmProjects\CetoniSP\data_files\2020-10-23 TriangleWaves 1Hz"
folder_20mL = r"G:\Shared drives\MSL - Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20201105 TriangleWaves 0.2mL"
folder_0p05 = r"G:\Shared drives\MSL - Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20201109 TriangleWaves 0.05mL"

folder = r'G:\Shared drives\MSL - Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20201209 Steps 0.75mL'

# folder = r"C:\Users\r.hawke\PycharmProjects\CetoniSP\data_files"
# filename = "Tri_0.05_5.2_1604875515.3838775_LVDT.xlsx"
#"Tri_0.05_2_1604610559.7766001_LVDT.xlsx"
#"Tri_0.05_3_1604610780.0006_LVDT.xlsx"
#"Tri_0.05_4_1604610994.7816_LVDT.xlsx"
#"Tri_0.05_5_1604611166.6785998_LVDT.xlsx"
#"Tri_0.05_6_1604611332.8676_LVDT.xlsx"
    #"Tri_0.05_7_1604611530.5206_LVDT.xlsx"
    # "Tri_0.05_9_1604611858.8866_LVDT.xlsx"
    # "Tri_0.05_8_1604611697.3955998_LVDT.xlsx"

# files = get_all_fnames(folder_0p05, "Tri_", endpattern="_LVDT.xlsx")
# #
filename = r'Step_0.75_25_1607473103.6122003_all.xlsx'
t_start, f0, Q, slope, offset, stdres = fit_damping(folder, filename, t_start=20.18, end=21.9)
print(t_start, f0, Q, slope, offset, stdres)

# filename = r'Tri_0.05_5.9_1604876796.7822_LVDT.xlsx'
# t_start, f0, Q, slope, offset = fit_damping(folder, filename, t_start=None)
# print(t_start, f0, Q, slope, offset)

# fout = load_workbook("DampedOsc.xlsx")
# sh = fout.active
# 0.007781219769354897
# for i in range(47):
#     filename = sh["A"+str(i+1)].value
#     print(filename)
#     t_start = float(sh["B"+str(i+1)].value)
#     t_start, f0, Q, slope, offset = fit_damping(folder, filename, t_start=t_start)
#     sh["C" + str(i + 1)] = f0
#     sh["D"+str(i+1)] = Q
#     sh["E"+str(i+1)] = slope
#     sh["F"+str(i+1)] = offset
# #     sh.append([filename, t_start, f0, Q, slope, offset])
# #
# fout.save("DampedOsc.xlsx")

