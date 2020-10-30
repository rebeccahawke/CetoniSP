import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows


def linear(x, a, b):
    return a * x + b

def fit_linear(x, y, a, b):

    plt.scatter(x, y)

    # Fit a quadratic to the data
    pars, cov = curve_fit(f=linear, xdata=x, ydata=y, p0=[a, b], bounds=(-np.inf, np.inf))

    print(pars)

    # Get the standard deviations of the parameters (square roots of the diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    print("Standard deviations: {}".format(stdevs))

    # Calculate the residuals
    res = y - linear(x, *pars)

    plt.scatter(x, linear(x, *pars), label='Fit: {:.3g} x + {:.3g}'.format(*pars))
    plt.title("Calibration for LVDT from RFCounter data")
    plt.xlabel("LVDT voltage (V)")
    plt.ylabel("Height (mm)")
    plt.legend()
    plt.show()


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def fit_quadratic(x, y, a, b, c):

    plt.scatter(x, y)

    # Fit a quadratic to the data
    pars, cov = curve_fit(f=quadratic, xdata=x, ydata=y, p0=[a, b, c], bounds=(-np.inf, np.inf))

    print(pars)

    # Get the standard deviations of the parameters (square roots of the diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    print("Standard deviations: {}".format(stdevs))

    # Calculate the residuals
    res = y - quadratic(x, *pars)

    plt.scatter(x, quadratic(x, *pars), label='Fit: {:.3g} x$^2$ + {:.3g} x + {:.3g}'.format(*pars))
    plt.title("Calibration for LVDT from RFCounter data")
    plt.xlabel("LVDT voltage (V)")
    plt.ylabel("Height (mm)")
    plt.legend()
    plt.show()


# filename = r'C:\Users\r.hawke\PycharmProjects\CetoniSP\10-1-5_osc.xlsx'
# A = 10
# T = 1
# cycles = 5
# phi = 0
# c = 0


def sinusoid(x, A, T, phi, c):
    return A * np.sin(2 * np.pi * x / T + phi) + c


def fit_sinusoid(filename, col, A, T, phi, c):

    df = pd.read_excel(filename)
    headers = list(df.columns.values.tolist())

    # plt.scatter(df[headers[0]], df[headers[1]], label=headers[1])
    plt.scatter(df[headers[0]], df[headers[col]], label=headers[col])

    x_fit = df[headers[0]]

    # Fit the sinusoidal data
    pars, cov = curve_fit(f=sinusoid, xdata=x_fit, ydata=df[headers[col]], p0=[A, T, phi, c], bounds=(-np.inf, np.inf))

    # Get the standard deviations of the parameters (square roots of the diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))

    # Calculate the residuals
    res = df[headers[col]] - sinusoid(x_fit, *pars)

    plt.plot(x_fit, sinusoid(x_fit, *pars), label='Fit')
    plt.legend()
    plt.show()


def plot_sp_rfc_data(SP_csv, RFC_csv, savepath=None):

    sp_data = pd.read_csv(SP_csv)
    rfc_data = pd.read_csv(RFC_csv)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    # fig.suptitle('Piston height over time')
    ax1.set_ylabel('Height (mm)')
    # ax1.set_title('From Capacitor')
    ax2.set_ylabel('Syringe fill level (mm)')
    ax2.set_xlabel('Timestamp')
    ax1.plot(pd.to_datetime(rfc_data['Timestamp']), rfc_data['Height (mm)'])
    ax2.plot(pd.to_datetime(sp_data['Timestamp']), sp_data['SP Position (mL)'])
    fig.autofmt_xdate()

    # plt.plot(pd.to_datetime(sp_data['Timestamp']), sp_data[' SP Position (mL)'])
    # plt.plot(pd.to_datetime(rfc_data['Timestamp']), rfc_data[" Frequency (Hz)"]/7500)
    plt.show()

    if savepath is not None:
        fig.savefig(os.path.join(r'../data_files',savepath))

        sp =os.path.join(r'../data_files', savepath.strip(".png")+".xlsx")

        wb = openpyxl.Workbook()
        sheet1 = wb.active
        sheet1.title = "SP data"
        for r in dataframe_to_rows(sp_data, index=True, header=True):
            sheet1.append(r)
        sheet2 = wb.create_sheet("RFC data")
        for r in dataframe_to_rows(rfc_data, index=True, header=True):
            sheet2.append(r)

        wb.save(filename=sp)


if __name__ == "__main__":
    rf = "RF-data_1603424127.0928001.csv"
    sp = "Tri_0.3_36_1603424129.4968002.csv"
    plot_sp_rfc_data(
        os.path.join(r'../data_files', sp),
        os.path.join(r'../data_files', rf),
        savepath=sp.strip(".csv")+".png"
    )
