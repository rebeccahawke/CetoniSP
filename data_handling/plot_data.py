import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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
