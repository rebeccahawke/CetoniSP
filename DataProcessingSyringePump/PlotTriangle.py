import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = r'I:\MSL\Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20190925'

data = pd.read_csv(path+'\\'+'20190925_cont_0.75mL_20sccm_20.0_1569368175_alldata.csv')

time_raw = data['Time (s)']
time = time_raw - time_raw[0]

raw_lvdt = data['Raw_LVDT']

height = raw_lvdt

plt.plot(time, height, marker='.', markersize=4)

plt.show()