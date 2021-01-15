import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

folder = r'G:\Shared drives\MSL - Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20201209 Steps 0.75mL'
file = r'Summary_step_fits_forPython.xlsx'

summary_data = pd.read_excel(os.path.join(folder, file))

nomflo = summary_data["flow (ccm)"]
piston_travel = summary_data["m (mm/s)"]

grpdflo = summary_data.groupby('flow (ccm)')["m (mm/s)"]

summary_data['VelAvg'] = grpdflo.transform('mean')
summary_data['VelMax'] = grpdflo.transform('max')
summary_data['VelMin'] = grpdflo.transform('min')
summary_data['VelRange'] = summary_data['VelMax'] - summary_data['VelMin']


fig, ax = plt.subplots(constrained_layout=True)

plt.errorbar(nomflo, summary_data['VelAvg'], summary_data['VelRange'], linestyle="", marker='o',)



plt.xlabel('Nominal flow rate of syringe plunger (ccm)')
plt.ylabel('Effective piston velocity (mm s$^{-1}$)')
# plt.savefig(path + '\\' + '20191002_step_summary1.png')


def vel2flo(x):
    # function to convert mm/s to ccm using the effective area of the piston of 196 mm2
    return x*196.12*0.001*60


def flo2vel(x):
    # function to convert mm/s to ccm using the effective area of the piston of 196 mm2
    return x/(196.12*0.001*60)


secaxy = ax.secondary_yaxis('right', functions=(vel2flo, flo2vel))
secaxy.set_ylabel('Effective flow at piston (ccm)')

# add identity line
xpoints = ax.get_xlim()
ypoints = [flo2vel(x) for x in xpoints]
ax.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False)

# Show the graph
# plt.legend()
# plt.title("Data from LVDT")
plt.show()