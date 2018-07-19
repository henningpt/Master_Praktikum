import matplotlib.pyplot as plt
import numpy as np
from tabelle import tabelle
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit


L = np.linspace(0,6)
r_1 = 1.4
r_2 = 1.4

plt.plot(L,(1-L/r_1)*(1-L/r_2) , '-' , label=r'g1g2_r_1=1.4')
r_1=1
plt.plot(L,(1-L/r_1)*(1-L/r_2) , '-' , label=r'g1g2_r_1=1')
plt.plot(L,(1)*(1-L/r_2) , '-' , label=r'g1g2_r_1=999999')

plt.hlines(1,0,2)
plt.hlines(0,0,2)

plt.legend(loc='best')
# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.tight_layout()
plt.savefig('build/plot_lange.pdf')

plt.close()


I = np.genfromtxt("messwerte_e.txt", unpack=True)
d_plot = np.linspace(-20, 30, 51)
plt.plot(d_plot, I, 'x')
plt.savefig('build/intensity.pdf')

plt.close()
I_polar = np.genfromtxt("polarisation.txt", unpack=True)
phi_plot = np.linspace(0, 360, 37)
plt.plot(phi_plot, I_polar, 'x')
plt.savefig('build/polarisation.pdf')
