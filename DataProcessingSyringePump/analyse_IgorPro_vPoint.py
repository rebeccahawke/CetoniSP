import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def linear(x, m, c):
     return m * x + c

# timeint = 0.05
path = r'C:\Users\r.hawke\Desktop\4OctFiles\SyringePumpTests\20191002\Igor_Pro_data'
# path = r'C:\Users\r.hawke\Desktop\4OctFiles\SyringePumpTests\20191004\Igor Pro data'

summary_file = pd.DataFrame()

t_params = pd.read_csv(path+'\\'+'t_params.csv', index_col='ID')

for f_i, filename in enumerate([f for f in os.listdir(path) if f.endswith('alldata.csv')]): # and f.startswith('20191002_set2')]):
    file_list = filename.split("_")
    set = file_list[1]
    flowrate = file_list[2].strip('r').strip('sccm')
    run = file_list[3]
    ID = set + flowrate + run
    # print(ID, t_params.loc[ID, 't1'])

    df = pd.read_csv(path+'\\'+filename)

    time = df["time_sec"]
    #height = df["syringe_mm"]
    height = df["syringe_mm"]

    i1 = int(t_params.loc[ID, 't1'])
    p1 = i1 - 10
    p2 = int(t_params.loc[ID, 't2'])
    p3 = int(t_params.loc[ID, 't3'])

    # curve fitting of leak rates
    linParams1, pcov1 = opt.curve_fit(linear, df.index[:p1], height[:p1])  # first leak rate
    linParams2, pcov2 = opt.curve_fit(linear, df.index[p3:], height[p3:])  # second leak rate
    linParams3, pcov3 = opt.curve_fit(linear, df.index[i1:p2], height[i1:p2])  # piston travel rate

    # Plot data and fits
    plt.plot(df.index[:], height[:], marker='o', markersize=3, label="Data")
    plt.plot(df.index[:p1], linear(df.index[:p1], *linParams1), label="Leak 1")
    plt.plot(df.index[p3:], linear(df.index[p3:], *linParams2), label="Leak 2")
    plt.plot(df.index[i1:p2], linear(df.index[i1:p2], *linParams3), label="Travel")

    # Show the graph
    plt.legend()
    plt.title(flowrate)
    plt.suptitle(ID + ' ' + str(p1) + ' ' + str(p2) + ' ' + str(p3))
    plt.xlabel('Point (#)')
    # plt.ylabel('Height (mm)')
    plt.ylabel('Syringe pump position (mm)')
    plt.savefig(path + '\\' + '20191002_step_' + flowrate + '_' + run + '_vPoints_SP.png')
    plt.show()
    plt.clf()

    isect1 = (linParams3[1] - linParams1[1]) / (linParams1[0] - linParams3[0])
    isect2 = (linParams2[1] - linParams3[1]) / (linParams3[0] - linParams2[0])

    summary_file = summary_file.append(
        {
            "ID": ID,
            "Flowrate (sccm)": flowrate,
            "Fall 1 m": linParams1[0],
            "u(Fall 1 m)": np.sqrt(pcov1[0][0]),
            "Fall 1 c": linParams1[1],
            "u(Fall 1 c)": np.sqrt(pcov1[1][1]),
            "Intersect 1": isect1,
            "Travel m": linParams3[0],
            "u(Travel m)": np.sqrt(pcov2[0][0]),
            "Travel c": linParams3[1],
            "u(Travel c)": np.sqrt(pcov2[1][1]),
            "Intersect 2": isect2,
            "Fall 2 m": linParams2[0],
            "u(Fall 3 m)": np.sqrt(pcov3[0][0]),
            "Fall 2 c": linParams2[1],
            "u(Fall 3 c)": np.sqrt(pcov3[1][1]),
        }, ignore_index=True)

summary_file.to_csv(path + '\\' + 'summary_data_vPoints_SP.csv')
