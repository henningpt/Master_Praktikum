import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
from scipy.signal import argrelextrema as rextrem
from uncertainties import unumpy as unp
import scipy.constants as con


# functions
def filter(arr1, arr2, cut):
    arr2_cut = arr2[abs(arr1) > cut]
    arr1_cut = arr1[abs(arr1) > cut]
    return(arr1_cut, arr2_cut)


def stplot(time, signal, logarit, name):
    ys_add = ""
    plt.plot(time, signal, 'x', label='Messwerte')
    plt.xlabel(r'$  t \ / \ \mathrm{s}$')
    if(logarit):
        ys_add = "ln "
    plt.ylabel(ys_add + r'$U \ / \ \mathrm{V}$')
    plt.legend(loc='best')
    plt.savefig("build/t_u_plot" + name + ".pdf")
    return()



def t1_bestimmung(t, a, T1):
    return(a * (1 - 2 * np.e**(-t/T1)))


def expf(x, a, b):
    return(a * np.e**(b * x))


def lnr(x, a, b):
    return(a * x + b)


def viskos(rho, alpha, t, sigma):
    return(rho * alpha *(t - sigma))


def radius(vis, diff):
    return(con.k * (25.0 + 273.15) / (6.0   * np.pi * vis * diff))


def grad(d, g, th):
    return(8.8 / (d * g * th))


def thalb(arr):
    halb_wert = max(arr) * 0.5
    arr = np.array(arr-halb_wert)
    return(np.argmin(abs(arr)))
