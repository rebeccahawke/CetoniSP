import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def linear(x, a=1, b=1):
    return [a * x_ + b for x_ in x]

def fit_linear(x, y, a=1, b=1):

    plt.scatter(x, y)

    # Fit a quadratic to the data
    pars, cov = curve_fit(f=linear, xdata=x, ydata=y, p0=[a, b], bounds=(-np.inf, np.inf))

    # print(pars)

    # Get the standard deviations of the parameters (square roots of the diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    # print("Standard deviations: {}".format(stdevs))

    # Calculate the residuals
    fit = linear(x, *pars)
    res = [y_ - x_ for y_, x_ in zip(y, fit)]

    plt.plot(x, linear(x, *pars), label='Fit: {:.3g} x + {:.3g}'.format(*pars))
    plt.scatter(x, res, )
    plt.title("Calibration for LVDT from RFCounter data")
    plt.xlabel("LVDT voltage (V)")
    plt.ylabel("Height (mm)")
    plt.legend()
    plt.show()
    plt.close()

    return pars


def quadratic(x, a=0.1, b=1, c=10):
    return [a * x_**2 + b * x_ + c for x_ in x]

def fit_quadratic(x, y, a=0.1, b=1, c=10):

    plt.scatter(x, y)

    # Fit a quadratic to the data
    pars, cov = curve_fit(f=quadratic, xdata=x, ydata=y, p0=[a, b, c], bounds=(-np.inf, np.inf))

    print(pars)

    # Get the standard deviations of the parameters (square roots of the diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    print("Standard deviations: {}".format(stdevs))

    # Calculate the residuals
    fit = quadratic(x, *pars)
    res = [y_ - x_ for y_, x_ in zip(y, fit)]

    plt.plot(x, quadratic(x, *pars), label='Fit: {:.3g} x$^2$ + {:.3g} x + {:.3g}'.format(*pars))
    plt.scatter(x, res)
    plt.title("Calibration for LVDT from RFCounter data")
    plt.xlabel("LVDT voltage (V)")
    plt.ylabel("Height (mm)")
    plt.legend()
    plt.show()

    return pars


def cubic(x, a=0.1, b=1, c=10., d=100):
    return [a * x_**3 + b * x_**2 + c * x_ + d for x_ in x]

def fit_cubic(x, y, a=0.1, b=1, c=10, d=100):

    plt.scatter(x, y)

    # Fit a quadratic to the data
    pars, cov = curve_fit(f=cubic, xdata=x, ydata=y, p0=[a, b, c, d], bounds=(-np.inf, np.inf))

    print(pars)

    # Get the standard deviations of the parameters (square roots of the diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    print("Standard deviations: {}".format(stdevs))

    # Calculate the residuals
    fit = cubic(x, *pars)
    res = [y_ - x_ for y_, x_ in zip(y, fit)]

    plt.plot(x, cubic(x, *pars), label='Fit: {:.3g} x$^3$ + {:.3g} x$^2$ + {:.3g} x + {:.3g}'.format(*pars))
    plt.scatter(x, res)
    plt.title("Calibration for LVDT from RFCounter data")
    plt.xlabel("LVDT voltage (V)")
    plt.ylabel("Height (mm)")
    plt.legend()
    plt.show()

    return pars

def quartic(x, a=0.1, b=1, c=10., d=100, e=1000):
    return [a * x_**4 + b * x_**3 + c * x_**2 + d * x_ + e for x_ in x]

def fit_quartic(x, y, a=0.1, b=1, c=10., d=100, e=1000):

    fig = plt.scatter(x, y, label="Raw data")

    # Fit a quadratic to the data
    pars, cov = curve_fit(f=quartic, xdata=x, ydata=y, p0=[a, b, c, d, e], bounds=(-np.inf, np.inf))

    print(pars)

    # Get the standard deviations of the parameters (square roots of the diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    print("Standard deviations: {}".format(stdevs))

    # Calculate the residuals
    fit = quartic(x, *pars)
    res = [y_ - x_ for y_, x_ in zip(y, fit)]

    plt.scatter(x, quartic(x, *pars), label='Fit: {:.3g} x$^4$ + {:.3g} x$^3$ + {:.3g} x$^2$ + {:.3g} x + {:.3g}'.format(*pars))
    plt.scatter(x, res, label="Residuals")
    plt.title("Calibration for RFCounter from dial gauge measurements")
    plt.xlabel("Height (mm)") #LVDT voltage (V)")
    plt.ylabel("RF Counter frequency (Hz)")
    plt.legend()
    plt.show()

    return pars

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





if __name__ == "__main__":
    rf = "RF-data_1603424127.0928001.csv"
    sp = "Tri_0.3_36_1603424129.4968002.csv"
    plot_sp_rfc_data(
        os.path.join(r'../data_files', sp),
        os.path.join(r'../data_files', rf),
        savepath=sp.strip(".csv")+".png"
    )
