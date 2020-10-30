"""
Extract data from a single ramp step
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def linear(x, m, c):
     return m * x + c

timeint = 0.05
path = r'I:\MSL\Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20191002'

summary_file = pd.DataFrame()

t_params = pd.read_csv(path+'\\'+'t_params_all.csv', index_col='Timestamp (s)')

# filename = '20191002_step_5sccm_w_5.0_1569975062_alldata.csv'
for filename in [f for f in os.listdir(path) if f.endswith('alldata.csv') and f.startswith('20191002_step')]:
    file_list = filename.split("_")
    flowrate = file_list[-3]
    timestamp = file_list[-2]

    df = pd.read_csv(path+'\\'+filename)

    time = df["Time (s)"]
    height = df["Height_rfc"]

    # get times for start and finish of travel and stabilisation

    ID = t_params.loc[int(timestamp), 'ID']
    t1 = t_params.loc[int(timestamp), 't1']
    t2 = t_params.loc[int(timestamp), 't2']
    t3 = t_params.loc[int(timestamp), 't3']

    p1 = int(t1 / timeint) - 10
    p2 = int(t2 / timeint)
    p3 = int(t3 / timeint)
    i1 = p1 + 10

    # curve fitting of leak rates
    linParams1, pcov1 = opt.curve_fit(linear, time[:p1], height[:p1])       # first leak rate
    linParams2, pcov2 = opt.curve_fit(linear, time[p3:], height[p3:])       # second leak rate
    linParamsT, pcovT = opt.curve_fit(linear, time[i1:p2], height[i1:p2])   # piston travel rate

    # Plot data and fits
    plt.plot(time[:], height[:], marker='o', markersize=3, label="Data")
    plt.plot(time[:p1], linear(time[:p1], *linParams1), label="Leak 1")
    plt.plot(time[p3:], linear(time[p3:], *linParams2), label="Leak 2")
    plt.plot(time[i1:p2], linear(time[i1:p2], *linParamsT), label="Travel")

    # Show the graph
    plt.legend()
    plt.title(flowrate)
    plt.suptitle(timestamp +' '+ str(t1) +' '+ str(t2) +' '+ str(t3))
    plt.xlabel('Time (s)')
    plt.ylabel('Height (mm)')
    # plt.savefig(path+'\\'+'20191002_step_' + flowrate + '_' + timestamp + '_CAP.png', dpi=600)
    # plt.show()
    plt.clf()

    isect1 = (linParamsT[1] - linParams1[1]) / (linParams1[0] - linParamsT[0])
    isect2 = (linParams2[1] - linParamsT[1]) / (linParamsT[0] - linParams2[0])

    plt.title(flowrate+' Residuals')
    # plt.plot(time[:p1], height[:p1] - linear(time[:p1], *linParams1), label="Leak 1")
    # plt.plot(time[p3:], height[p3:] - linear(time[p3:], *linParams2), label="Leak 2")
    plt.plot(time[i1:p2], height[i1:p2] - linear(time[i1:p2], *linParamsT), label="Travel")
    # plt.show()
    # plt.savefig(path+'\\'+'20191002_step_' + flowrate + '_' + timestamp + '_CAP_resTrav.png', dpi=600)
    plt.clf()

    summary_file = summary_file.append(
        {
            "Timestamp (s)": timestamp,
            "ID": ID,
            "Flowrate (sccm)": flowrate,
            "Fall 1 m": linParams1[0],
            "u(Fall 1 m)": np.sqrt(pcov1[0][0]),
            "Fall 1 c": linParams1[1],
            "u(Fall 1 c)": np.sqrt(pcov1[1][1]),
            "Intersect 1": isect1,
            "Travel m": linParamsT[0],
            "u(Travel m)": np.sqrt(pcovT[0][0]),
            "Travel c": linParamsT[1],
            "u(Travel c)": np.sqrt(pcovT[1][1]),
            "Intersect 2": isect2,
            "Fall 2 m": linParams2[0],
            "u(Fall 2 m)": np.sqrt(pcov2[0][0]),
            "Fall 2 c": linParams2[1],
            "u(Fall 2 c)": np.sqrt(pcov2[1][1]),
        }, ignore_index=True)

summary_file.to_csv(path+'\\'+'summary_data_CAP.csv')






