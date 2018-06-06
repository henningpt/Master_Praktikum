import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
from scipy.signal import argrelextrema as rextrem
from uncertainties import unumpy as unp

# werte
tau   = 20e-3
phi   = 111 # degree
f     = 21.71617e6


# lade daten
t1_visk = unp.uarray(12.0 * 60.0 + 3.0 + 0.5, 1)
t2_visk = unp.uarray(12.0 * 60.0 + 0.3, 1)  # zeiten viskosimeter

t1_tau, t1_amp = np.genfromtxt("t1messung.txt", unpack=True)
d_tau, d_amp = np.genfromtxt("diffmessung.txt", unpack=True)
# d_amp *= -1.0 keine ahnung wofuer :D

t_visk = 0.5 * (t1_visk + t2_visk)
print("mittelwert viskosimeter: ", t_visk)


# lade csv dateien:
mydata = np.genfromtxt('cp.csv', delimiter=',')
time_a = mydata[:,0]
sig_a  = mydata[:,1]

mydata2 = np.genfromtxt('mg.csv', delimiter=',')
time_a2 = mydata2[:,0]
# zeitskala verschieben, sodass keine negativen Zeiten auftreten
time_a2 += abs(min(time_a2))
sig_a2  = mydata2[:,1]


# functions
def filter(arr1, arr2, cut):
    arr2_cut = arr2[abs(arr1) > cut]
    arr1_cut = arr1[abs(arr1) > cut]
    return(arr1_cut, arr2_cut)


def stplot(time, signal, logarit, name):
    ys_add = ""
    plt.plot(time, signal, 'x', label='Messwerte')
    plt.xlabel(r'$  t \ / \ \mathrm{ms}$')
    if(logarit):
        ys_add = "ln "
    plt.ylabel(ys_add + r'$U \ / \ \mathrm{mV}$')
    plt.legend(loc='best')
    plt.savefig("build/t_u_plot" + name + ".pdf")
    return()


def expf(x, a, b, c):
    return(a * np.e**(b * x) + c)


def lnr(x, a, b):
    return(a * x + b)


def diffkoeff(t, a, T2, D, g, G):
    return(a * np.e**(-t / T2) * np.e**(- D * g**2 * G**2 * t**3 / 12.0))


def grad(d, g, th):
    return(8.8 / (d * g * th))


def thalb(arr):
    halb_wert = max(arr) * 0.5
    arr += -halb_wert
    return(np.argmin(abs(arr)))


# verarbeiten
#cutten
sig_a_cut , time_a_cut = filter(sig_a, time_a, 21e-3)
sig_a2_cut, time_a2_cut = filter(sig_a2, time_a2, 0e-2)

# peaks suchen
extrema_a = rextrem(sig_a_cut, np.greater, order=4)
extrema_a2 = rextrem(sig_a2_cut, np.greater, order=29)


# fits, rechnungen
# a)
# aparams1, acov1 = cf(expf, time_a2_cut[extrema_a2], sig_a2_cut[extrema_a2], maxfev=2000) curve_fit klappt nicht

# log -> linfit
aparams, acov = cf(lnr, time_a2_cut[extrema_a2], np.log(sig_a2_cut[extrema_a2]))
auparams = unp.uarray(aparams, np.sqrt(np.diag(acov)))

t2_a = -1 / auparams[0] # berechne T2 fuer meiboom-gill methode

# fit fuer halbwertszeit
dparams, dcov = cf(diffkoeff, d_tau, d_amp)
duparams = unp.uarray(dparams, np.sqrt(np.diag(dcov)))


# plotten
stplot(d_tau, d_amp, 0, 'diff')
d_plot = np.linspace(min(d_tau), max(d_tau))
plt.errorbar(d_plot, diffkoeff(d_plot, *dparams), label='Fit', fmt='-')
plt.savefig("build/t_u_plotdiff.pdf")
plt.close()
stplot(time_a, sig_a, 0, "1")
plt.close()
stplot(time_a_cut, sig_a_cut, 0, '1cut')
plt.close()
stplot(time_a2, sig_a2, 0, "2")
plt.close()
stplot(time_a2_cut, sig_a2_cut, 0, "2cut")
plt.close()
stplot(time_a2_cut[extrema_a2], np.log(sig_a2_cut[extrema_a2]), 1, "2_extrem")
t_plot = np.linspace(min(time_a2_cut[extrema_a2]), max(time_a2_cut[extrema_a2]))
plt.errorbar(t_plot, lnr(t_plot, *aparams), yerr=0, label='Fit', fmt='-')
plt.savefig("build/t_u_plot" + "2_extrem" + ".pdf")
plt.close()


# berechne halbwertszeit aus fit:
t_half_exact = d_plot[thalb(diffkoeff(d_plot, *dparams))]


# ausgeben
print("\nT2 berechnet: ", t2_a)
print("params aus cf: ", auparams)
print(d_tau[thalb(d_amp)])
print("exaktwert : ", t_half_exact)
print("\n\nparams from diff_fit: ")
for enum in enumerate(duparams):
    print("\n", enum)
