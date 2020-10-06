import numpy
from numpy.linalg import inv, multi_dot

""" This code uses matrix least squares analysis for circular weighing measurement sequences
Required inputs:
    q = number of independent items being compared
    h = highest order of drift correction
    y_col = column vector of measured values or comparator readings in order taken
    t1 = row vector of times for each measurement
Function outputs:
    The design matrix 'X', a column vector 'b' of expected values, 
    and its variance-covariance matrix 'C'
    Estimates of item differences and their standard deviations
    Drift parameters and their standard deviations 
For more information, see 'A General Approach to Comparisons in the Presence of Drift'
"""


def design_matrix(q,h,t1):
    p = t1.size
    # Prepare matrices of times squared (quadratic drift) and times cubed (cubic drift)
    t2 = numpy.multiply(t1,t1)
    t3 = numpy.multiply(t2,t1)

    # Prepare Design Matrix X
    id = numpy.identity(q)
    i = 1
    idp = id
    while i<(p/q):
        idp = numpy.concatenate([idp, id], axis=1)
        i += 1

    X_T = numpy.vstack((idp, t1))
    if h > 1:
        X_T = numpy.vstack((X_T, t2))
    if h > 2:
        X_T = numpy.vstack((X_T, t3))
    X = X_T.T
    #print('Design Matrix, X = ')
    #print(X)

    return X


def expected_values(X,y_col):
    X_T = X.T
    # Calculate the expected values:
    XtXinv = inv(numpy.dot(X_T,X))
    # print('XtXinv = ')
    # print(XtXinv)

    b = numpy.linalg.multi_dot([XtXinv,X_T,y_col])
    #print('b = ')
    #print(b)

    return b


def varcovarmatrix(X,y_col,b,q,h):
    # calculate the residuals, variance and variance-covariance matrix:
    residuals = y_col - numpy.dot(X, b)
    #print('residuals = ')
    #print(residuals)

    p = len(y_col)
    v = p - q - h  # degrees of freedom
    #print('degrees of freedom = ', v)

    var = numpy.dot(residuals.T,residuals)/(v)
    #print('variance, \u03C3\u00b2 = ',var.item(0))
    #print('standard deviation, \u03C3 = ',numpy.sqrt(var.item(0)))

    XtXinv = inv(numpy.dot(X.T,X))
    C = numpy.multiply(var,XtXinv)
    #print('variance-covariance matrix, C = ')
    #print(C)
    #print('for ',q,' item(s), and ',h,' drift coefficient(s)')

    return C


def item_diff(item1, item2, q, h, b, C):
    # Calculate desired differences and/or drift coefficients:
    w_T = numpy.zeros(1, q+h) # row vector
    w_T[0,item1-1] = 1
    w_T[0, item2-1] = -1
    #print(w_T)
    w = numpy.vstack(w_T) # column vector
    diffab = numpy.dot(w_T,b).item(0)
    vardiffab = numpy.linalg.multi_dot([w_T,C,w]).item(0)
    stdev_diffab = numpy.sqrt(vardiffab)

    print('Difference between selected items ( stdev ): ', diffab, '(',stdev_diffab,')')

    return [diffab, stdev_diffab]


def drift_coeffs(q, h, b, C):
    driftcoeffs = numpy.matrix(numpy.zeros((h, 2)))
    driftcoeffs[:, 0] = b[q:q + h]

    d = numpy.diagonal(C)
    i = 0
    while i < h:
        driftcoeffs[i, 1] = numpy.sqrt(d[i + q])
        i += 1

    print('Matrix of drift coefficients and their standard deviations:')
    print(driftcoeffs)

    return driftcoeffs


"""Call the functions below to return item differences and drift coefficients"""

def matrixleastsquares_itemdiff(q,h,y_col,t1,item1,item2):
    X = design_matrix(q, h, t1)
    b = expected_values(X, y_col)
    C = varcovarmatrix(X, y_col, b, q, h)
    return item_diff(item1, item2, q, h, b, C)
    # returns a 1x2 row vector: [difference item1 - item2, std deviation]


def matrixleastsquares_driftcoeffs(q,h,y_col,t1):
    X = design_matrix(q, h, t1)
    b = expected_values(X, y_col)
    C = varcovarmatrix(X, y_col, b, q, h)
    return drift_coeffs(q, h, b, C)
    # returns a hx2 matrix: [drift coefficients, std deviations]