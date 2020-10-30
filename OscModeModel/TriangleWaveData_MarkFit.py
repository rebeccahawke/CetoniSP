import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

msl_orange = '#f15a29'

h = 3 #6.26/2.54
w = 4 #7.98/2.54

datapath = r'I:\MSL\Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\Damped oscillator response\Modelling of triangle wave response'
savepath = r'I:\MSL\Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\Damped oscillator response\Modelling of triangle wave response'

data_5sccm = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_5sccm_092_data.csv', header=None)
model_5sccm = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_5sccm_092_model.csv', header=None)
data_10sccm = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_10sccm_539_data.csv', header=None)
model_10sccm = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_10sccm_539_model.csv', header=None)
data_15sccm = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_15sccm_892_data.csv', header=None)
model_15sccm = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_15sccm_892_model.csv', header=None)
data_20sccm = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_20sccm_273_data.csv', header=None)
model_20sccm = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_20sccm_273_model.csv', header=None)
data_25sccm = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_25sccm_549_data.csv', header=None)
model_25sccm = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_25sccm_549_model.csv', header=None)
data_30sccm = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_30sccm_441_data.csv', header=None)
model_30sccm = pd.read_csv(datapath + '\\' + '20190925_cont_0.75mL_30sccm_441_model.csv', header=None)

data_30sccm[0] = data_30sccm[0] - data_30sccm[0][14]
model_30sccm[0] = model_30sccm[0] - model_30sccm[0][40]

fig, axes = plt.subplots(4, 1,  sharex='col', figsize=[w, h*2])

ax5  = axes[0]
ax15 = axes[1]
ax20 = axes[2]
ax30 = axes[3]

# ax5 =  axes[0, 0]
# ax10 = axes[1, 0]
# ax15 = axes[2, 0]
# ax20 = axes[0, 1]
# ax25 = axes[1, 1]
# ax30 = axes[2, 1]

xmax = 20
y = 2.5

for ccm in [5, 15, 20, 30]:
    # print(i, 5*(i+1), ax)
    ax = eval('ax'+str(ccm))
    data = eval('data_'+str(ccm)+'sccm')
    model = eval('model_'+str(ccm)+'sccm')
    ax.plot(data[0], 1000 * data[1], label="Data", marker='.', markersize=4, color=msl_orange, linestyle='None')
    ax.plot(model[0], 1000 * model[1], label='Model', color='k', alpha=0.5)
    ax.set_xlim(0, xmax)

ax20.set_ylabel('Piston position (mm)')

ax5.legend()
# ax15.set_xlabel('Time (s)')
ax30.set_xlabel('Time (s)')

plt.subplots_adjust(hspace=.0)

figname = "TriangleData_MarkModel_panels_5-15-20-30"

fig.savefig(savepath + '\\' + figname + '.png', dpi=1200)
fig.savefig(savepath + '\\' + figname + '.svg')

plt.show()