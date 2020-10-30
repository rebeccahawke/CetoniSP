import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

path = r'I:\MSL\Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20191002_fit2'

summary_data = pd.read_csv(path+'\\'+'summary_data_allflowrates.csv', index_col='Timestamp (s)')

nomflo = summary_data["Flowrate (sccm)"]
piston_travel = summary_data["Travel m"]

plt.scatter(nomflo, piston_travel)

# Show the graph
# plt.legend()
plt.title("Data from capacitor")

plt.xlabel('Nominal flow rate of syringe pump (sccm)')
plt.ylabel('Piston velocity (mm/s)')
plt.savefig(path + '\\' + '20191002_step_summary1.png')
plt.show()