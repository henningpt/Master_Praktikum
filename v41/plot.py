import matplotlib.pyplot as plt
import numpy as np


# lade daten
radien_glas = np.genfromtxt("v41_glas.txt", unpack=True)
radien_meta = np.genfromtxt("v41_metall.txt", unpack=True)
radien_salz = np.genfromtxt("v41_salz.txt", unpack=True)


# groessen definie.txren
wavelen =
camera_rad = 57.3 * 10**(-3)
dist_probe = 130 * 10**(-3)


# functions

# x = np.linspace(0, 10, 1000)
# y = x ** np.sin(x)
#
# plt.subplot(1, 2, 1)
# plt.plot(x, y, label='Kurve')
# plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
# plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
# plt.legend(loc='best')
#
# plt.subplot(1, 2, 2)
# plt.plot(x, y, label='Kurve')
# plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
# plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
# plt.legend(loc='best')
#
# # in matplotlibrc leider (noch) nicht möglich
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('build/plot.pdf')
