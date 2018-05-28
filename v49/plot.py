import numpy as np
import matplotlib.pyplot as plt
# import scipy.signal.argrelextrema as rextrem
from scipy.signal import argrelextrema as rextrem

# lade daten
t1_visk = 12.0 * 60.0 + 3.0 + 0.5
t2_visk = 12.0 * 60.0 + 0.3  # zeiten viskosimeter

t1_tau, t1_amp = np.genfromtxt("t1messung.txt", unpack=True)
d_tau, d_amp = np.genfromtxt("diffmessung.txt", unpack=True)
d_amp *= -1.0

print("mittelwert viskosimeter: ", 0.5 * (t1_visk + t2_visk))

# lade csv dateien:
mydata = np.genfromtxt('cp.csv', delimiter=',')
time_a = mydata[:,0]
sig_a  = mydata[:,1]

mydata2 = np.genfromtxt('mg.csv', delimiter=',')
time_a2 = mydata2[:,0]
sig_a2  = mydata2[:,1]

# filtern auf signifikante signale
# cut_here = 4e-2
# sig_a_cut = sig_a[abs(sig_a) > cut_here]
# time_a_cut = time_a[abs(sig_a) > cut_here]
# print(sig_a_cut)


# functions
def filter(arr1, arr2, cut):
    arr2_cut = arr2[abs(arr1) > cut]
    arr1_cut = arr1[abs(arr1) > cut]
    return(arr1_cut, arr2_cut)


def stplot(time, signal, logarit, name, *args):
    plt.plot(time, signal, '-', label='Messwerte')
    if (!args.empty()):
        plt.plot()
    plt.xlabel(r'$  t \ / \ \mathrm{ms}$')
    plt.ylabel(r'$U \ / \ \mathrm{mV}$')
    if(logarit):
        plt.xscale("log")
    plt.legend(loc='best')
    plt.savefig("build/t_u_plot" + name + ".pdf")
    plt.close()
    return()


# def findmax(arr, range):

# verarbeiten
#cutten
sig_a_cut , time_a_cut = filter(sig_a, time_a, 4e-2)
sig_a2_cut, time_a2_cut = filter(sig_a2, time_a2, 2e-2)

# peaks suchen
extrema_a = rextrem(sig_a, np.greater, order=3)

# # t1 plotten
# plt.figure(1)
# plt.plot(t1_tau, t1_amp, 'x', label='Messwerte')
# plt.xlabel(r'$  \tau \ / \ \mathrm{ms}$')
# plt.ylabel(r'$U \ / \ \mathrm{mV}$')
# # plt.xscale("log")
# plt.legend(loc='best')
# plt.show()
#
# # plotten diffusion
# plt.figure(2)
# plt.plot(d_tau, d_amp, 'x', label='Messwerte')
# plt.xlabel(r'$\tau \ / \ \mathrm{ms}$')
# plt.ylabel(r'$U \ / \ \mathrm{mV}$')
# plt.legend(loc='best')
# plt.show()


# plotten fuer a)
# plt.figure(3)
# plt.plot(time_a, sig_a, '-', label='Messwerte')
# plt.xlabel(r'$  t \ / \ \mathrm{ms}$')
# plt.ylabel(r'$U \ / \ \mathrm{mV}$')
# # plt.xscale("log")
# plt.legend(loc='best')
# plt.show()

# testen
testarr = np.array([1,0,1,1,2,1,2,0,3,1,2,1,0])
extrema = rextrem(testarr, np.greater, order=2)
print("TEST: " ,testarr, "\n", extrema, "\n", testarr[extrema])
# plotten
stplot(time_a, sig_a, 0, "1")
stplot(time_a_cut, sig_a_cut, 0, '1cut')
stplot(time_a2, sig_a2, 0, "zwei")
stplot(time_a2_cut, sig_a2_cut, 0, "2cut")
