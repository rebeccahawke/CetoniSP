import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import scipy.optimize as opt


def linear(x, m, c):
    return m * x + c


def dampedHO(t, A, omega0, Q, psi, slope, offset):
    gamma = omega0 / Q        # here gamma = Gamma/m; this relation is true for a weakly damped oscillator
    # omega = np.sqrt(omega0 ** 2 - (gamma/2) ** 2)
    omega = omega0 * np.sqrt(1 - 1/(4 * Q**2))
    return A * np.exp(-gamma / 2 * t) * np.sin(omega * (t - psi)) + slope * t + offset


msl_orange = '#f15a29'

h = 3
w = 4

path = r'G:\Shared drives\MSL - Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20201209 Steps 0.75mL'

dir = 1 # rise
# dir = -1  # fall
if dir > 0:
    p1 = os.path.join(path, 'Step_0.75_25_1607473103.6122003_all.xlsx')
    start = 20.179  # start time for ramp
    end = start + .75 / 25 * 60
    t_off = 0.25
else:
    p1 = os.path.join(path, 'Step_-0.75_25_1607473015.912431_all.xlsx')
    start = 20.1  # start time for ramp
    # end = 21.835  # end time for ramp
    end = start + .75/25*60
    t_off = 0.3

# extract SP data
dataSP1 = pd.read_excel(p1, 'SP data')
dataSPT1 = dataSP1["Timestamp"][1:]
dataSPP1 = dataSP1["SP Position (mL)"][1:]
# time_delta = datetime.strptime(dataSPT1[2], '%Y-%m-%d %H:%M:%S.%f') - datetime.strptime(dataSPT1[1], '%Y-%m-%d %H:%M:%S.%f')
# time_int = time_delta.total_seconds()

# np.linspace(0, time_int * len(dataSPT1), len(dataSPT1))

data1 = pd.read_excel(p1, 'RFC data')
timestamps1 = data1['Timestamp']

# curve fitting of rates - infusion (1)
time1 = data1['Time (s)']
height1 = data1['LVDT (mm)']
timeint = time1[2] - time1[1]

# convert to measurement index
start_int = int(start / timeint) + 2
end_int = int(end / timeint) + 2

# find time difference for 'time zero'
t0 = datetime.strptime(data1['Timestamp'][start_int], '%Y-%m-%d %H:%M:%S.%f')

t_sp = [t for t in dataSPT1]
for i, t in enumerate(dataSPT1):
    try:
        dtt = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
    except TypeError:  # in case for some bizarre reason it's already a datetime object
        dtt = t
    t_sp[i] = (dtt - t0).total_seconds() - t_off  # not sure why it's not quite aligned - time delay for sending back data?

SPP = [p - min(dataSPP1) for p in dataSPP1]

### do fitting
# piston fall rate
linParamsFall1, pcovFall1 = opt.curve_fit(linear, time1[1:start_int], height1[1:start_int])
linResFall1 = [y_ - x_ for y_, x_ in zip(height1[1:start_int], linear(time1[1:start_int], *linParamsFall1))]
linStdResFall1 = np.std(linResFall1, ddof=1)
print(linStdResFall1)

# piston travel rate - linear fit
linParamsRamp1, pcovLinRamp1 = opt.curve_fit(linear, time1[start_int:end_int]-start, height1[start_int:end_int])

linResRamp1 = [y_ - x_ for y_, x_ in zip(height1[start_int:end_int], linear(time1[start_int:end_int]-start, *linParamsRamp1))]
linStdResRamp1 = np.std(linResRamp1, ddof=1)
print(linStdResRamp1)

# piston travel rate - DHO fit
hoParamsRamp1, pcovHORamp1 = opt.curve_fit(dampedHO, time1[start_int:end_int]-start, height1[start_int:end_int],
                                                 p0=[1, 1, 7, 0, 2.3, np.average(height1[start_int:end_int])],
                                                 bounds=(-100, 100))

hoResRamp1 = [y_ - x_ for y_, x_ in zip(height1[start_int:end_int], dampedHO(time1[start_int:end_int]-start, *hoParamsRamp1))]
hoStdResRamp1 = np.std(hoResRamp1, ddof=1)
print(hoStdResRamp1)

# piston oscillation post ramp
hoParamsFall1, pcovHOFall1 = opt.curve_fit(dampedHO, time1[end_int:]-end, height1[end_int:],
                                         p0=[1, 1, 7, 0, -0.0001, np.average(height1[end_int:])],
                                         bounds=(-100, 100))
hoResFall1 = [y_ - x_ for y_, x_ in zip(height1[end_int:], dampedHO(time1[end_int:]-end, *hoParamsFall1))]
hoStdResFall1 = np.std(hoResFall1, ddof=1)
print(hoStdResFall1)

### make plots
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=[w, h*2], sharex=True, gridspec_kw={'hspace': 0})
# gridspec_kw={'hspace': 0} reduces the vertical space between subplots to zero

ax0.scatter(t_sp, SPP, color=msl_orange, s=2)
ax0.axvspan(0, end - start, alpha=0.5, color='silver')
ax0.set_ylabel("Syringe volume (mL)")

# plot raw data
ax1.plot(time1[:] - start, height1[:],
         label="Data", marker='.', color=msl_orange, linestyle='None')

# plot fits
ax1.plot(time1[1:start_int] - start, linear(time1[1:start_int], *linParamsFall1),
         color='k', linestyle='--')
# label="Linear fit with gradient = " + str(round(linParamsFall1[0], 4)) + "\n(u = " + str(
#             round(pcovFall1[0][0], 4)) + ") mm/s")
# ax1.plot(time1[start_int:end_int] - start, dampedHO(time1[start_int:end_int]-start, *hoParamsRamp1),
#          color='k', linestyle='--',
#          label="Fit based on model; \ntravel slope of {:.2f} \n(u = {:.2e}) mm/s" .format(hoParamsRamp1[4], pcovHORamp1[4][4]))
ax1.plot(time1[start_int:end_int] - start, linear(time1[start_int:end_int]-start, *linParamsRamp1),
         color='k', linestyle='--',
         label="Fit; travel slope\nof {:.3f} mm/s" .format(linParamsRamp1[0], pcovLinRamp1[0][0]))
 #            round(pcovHORamp1[4][4], 4)) + ") mm/s"")
#         label="Damped HO with gradient = " + str(round(hoParamsRamp1[4], 4)) + "\n(u = " + str(
 #            round(pcovHORamp1[4][4], 4)) + ") mm/s")
ax1.plot(time1[end_int:] - start, dampedHO(time1[end_int:]-end, *hoParamsFall1),
         color='k', linestyle='--',)
#         label="Damped HO fit")


ax1.set_ylabel('Piston position (mm)')  # Piston position
ax1.set_xlim(-10, 20)
# ax1.set_ylim(-0.5, 5.9)
# ax1.text(-0.15, 1.1, '(a)', transform=ax1.transAxes, size=12, )
ax1.axvspan(0, end - start, alpha=0.5, color='silver')
ax1.legend()

# plot residuals and damped
ax2.plot(time1[1:start_int] - start, linResFall1,
         color='k', label="Residual to fit")
ax2.plot(time1[start_int:end_int] - start, linResRamp1,
         color='k')
ax2.plot(time1[end_int:] - start, hoResFall1,
         color='k')

# ax2.plot(time1[start_int:end_int] - start, dampedHO(time1[start_int:end_int]-start, *hoParamsRamp1[:4], 0, 0),
#          color=msl_orange, label="Model DHO only")
ax2.plot(time1[end_int:] - start, dampedHO(time1[end_int:]-end, *hoParamsFall1[:4], 0, 0),
         color=msl_orange, label="DHO")
ax2.axvspan(0, end - start, alpha=0.5, color='silver')

ax2.set_ylabel('Position (mm)')  # Piston position
ax2.set_xlabel('Time (s)')
ax2.legend()

# add labels to panels
for ax, lab in zip([ax0, ax1, ax2], ["(i)", "(ii)", "(iii)"]):
    ax.text(-0.2, 1, lab, transform=ax.transAxes,
          style='italic', fontsize=12, va='top', ha='right')

if dir > 0:     # add labels for regions for rise
    ax1.text(-9, 1, 'Initial fall', style='italic')
    # ax1.text(0, 0.3, 'Tr.')
    ax1.text(5, 3, 'DHO', style='italic')

else:           # add labels for regions for drop
    ax1.text(-9, 3, 'Initial fall', style='italic')
    # ax1.text(0, 0.3, 'Tr.')
    ax1.text(5, 1, 'DHO', style='italic')

plt.tight_layout()
plt.show()
