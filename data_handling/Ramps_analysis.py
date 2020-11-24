"""
Analysis of single steps to evaluate reproducibility of syringe pump.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import scipy.optimize as opt

from data_handling.process_files import get_all_fnames, read_in_data
from data_handling.plot_data import linear, fit_linear


msl_orange = '#f15a29'


def separate_waits_from_ramp(sp_data, rfc_data):
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
        if 14.8 < p < 15:  # range of plunger movement
            ramp_is.append(sp_data["Timestamp"][i])

    # separate data for waits and ramp
    wait_pre = []
    ramp = []
    wait_post = []

    height_ts = rfc_data["Timestamp"][1:]
    for i, t in enumerate(height_ts):
        if t < min(ramp_is):
            wait_pre.append(i)
        elif t > max(ramp_is):
            wait_post.append(i)
        else:
            ramp.append(i)

    return min(ramp), max(ramp)



fol_steps = r'G:\Shared drives\MSL - Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20201124_Steps_0p5mL'

files = get_all_fnames(fol_steps, "Step_", endpattern="_LVDT.xlsx")

for fname in files:
    name_parts = fname.split("_")
    vol = float(name_parts[1])
    flo = float(name_parts[2])
    ts = name_parts[3]
    time_int = 0.02

    sp_data, rfc_data = read_in_data(fol_steps, fname)

    x = rfc_data["Timestamp"][1:]
    y = rfc_data["LVDT (mm)"][1:]
    x_i = range(len(y))
    plt.plot(y)
    plt.show()

    start = int(input("start"))
    num_pts = int(abs(vol)/flo*60/time_int)

    # start, end = separate_waits_from_ramp(sp_data, rfc_data)
    # print(fname, start, end)

    x_t = np.linspace(0, 0.02 * num_pts, num_pts)
    pars = fit_linear(x_t, y[start:start+num_pts])

    print(fname, start, num_pts, vol/0.5, flo, *pars)


    # if vol < 0:  # aspiration/withdrawal: plunger ends at 15 mL
    # elif vol > 0:  # dispensing/infusion: plunger starts at 15 mL


