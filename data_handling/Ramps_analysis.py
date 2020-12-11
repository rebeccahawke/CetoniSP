"""
Analysis of single steps to evaluate reproducibility of syringe pump.
"""
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import scipy.optimize as opt

from msl.qt import prompt

from data_handling.process_files import get_all_fnames, read_in_data
from data_handling.plot_data import linear, fit_linear


msl_orange = '#f15a29'


def separate_waits_from_ramp(sp_data, rfc_data, vol):
    """Determine which part of the data set contains the ramp

    Parameters
    ----------
    folder
    fname

    Returns
    -------
    tuple of start and end i for ramp.  Note that this i is for rfc_data["Timestamp"][1:]
    """


    # determine start and end times for ramp
    sp_pos = sp_data["SP Position (mL)"][1:]
    ramp_is = []
    for i, p in enumerate(sp_pos):
        if vol < 0 and 15.01 < p < 15.74:  # range of plunger movement for aspiration (-ve vol)
            ramp_is.append(datetime.strptime(sp_data["Timestamp"][i], '%Y-%m-%d %H:%M:%S.%f'))
        elif vol > 0 and 14.96 < p < 15.69:  # range of plunger movement for dispense (+ve vol)
            ramp_is.append(datetime.strptime(sp_data["Timestamp"][i], '%Y-%m-%d %H:%M:%S.%f'))

    # separate data for waits and ramp
    wait_pre = []
    ramp = []
    wait_post = []

    height_ts = rfc_data["Timestamp"][1:]
    for i, t in enumerate(height_ts):
        if datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') < min(ramp_is):
            wait_pre.append(i)
        elif datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') > max(ramp_is):
            wait_post.append(i)
        else:
            ramp.append(i)

    return min(ramp), max(ramp)



fol_steps = r"G:\Shared drives\MSL - Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20201209 Steps 0.75mL"

files = get_all_fnames(fol_steps, "Step_", endpattern="_all.xlsx")

summary2 = open("starts.csv", mode='a')

summary = "Summary1.xlsx"
f1 = os.path.join(fol_steps, summary)
xls = pd.ExcelFile(f1)
starts = pd.read_excel(xls, 'Sheet1')

for i, fname in enumerate(starts['fname']):
# for i, fname in enumerate(files):
    name_parts = fname.split("_")
    vol = float(name_parts[1])
    flo = float(name_parts[2])
    ts = name_parts[3]

    sp_data, rfc_data = read_in_data(fol_steps, fname)

    x = rfc_data["Timestamp"][1:]
    time_delta = datetime.strptime(x[2], '%Y-%m-%d %H:%M:%S.%f')-datetime.strptime(x[1], '%Y-%m-%d %H:%M:%S.%f')
    time_int = time_delta.total_seconds()
    x_i = np.linspace(0, time_int * len(x), len(x))

    ### Manually determine beginning of ramp ###
    y = rfc_data["LVDT (mm)"][1:]
    # plt.plot(y)
    # plt.show()
    # start = int(input("start"))
    # plt.clf()

    # num_pts = int(abs(vol) / flo * 60 / time_int)
    start = int(starts['start'][i])  # + wait
    end = int(starts['end'][i])
    # start, end = separate_waits_from_ramp(sp_data, rfc_data, vol)
    # print(fname, start, end)

    while True:
        num_pts = end - start
        wait_pts = int(20 / time_int)  # 20 s fall rate measurement before ramp
        if wait_pts > start:
            wait_pts = start
        wait_post = int(0 / time_int)  # allow some time for bounce to settle after ramp

        # Initial fall rate
        pars_pre, res_pre = fit_linear(x_i[0:wait_pts], y[0:wait_pts])

        # Ramp rate
        pars, res = fit_linear(x_i[start:start+num_pts], y[start:start+num_pts])

        # Final fall rate
        pars_post, res_post = fit_linear(x_i[start+num_pts+wait_post:], y[start+num_pts+wait_post:])

        plt.show()

        a = prompt.yes_no("Is this fitting satisfactory?")
        if a:
            break

        start = int(input("new start (was {})".format(start)))
        end = int(input("new end (was {})".format(end)))

    deltaH = [(pars_post[0] - pars_pre[0]) * x + pars_post[1] - pars_pre[1] for x in x_i]
    dh_ave = deltaH[start+int(num_pts/2)]
    dh_std_dev = np.std(deltaH[start:start+num_pts], ddof=1)

    print(fname, start, num_pts, vol/abs(vol), flo, *pars, res, *pars_pre, res_pre, *pars_post, res_post, dh_ave, dh_std_dev)
    summary2.writelines("{},{},{}\n".format(fname, start, end))

summary2.close()

    # plt.plot(x_i, deltaH)
    # plt.show()

    # if vol < 0:  # aspiration/withdrawal: plunger ends at 15 mL
    # elif vol > 0:  # dispensing/infusion: plunger starts at 15 mL


