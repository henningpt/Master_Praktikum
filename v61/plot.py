import matplotlib.pyplot as plt
import numpy as np
from tabelle import tabelle
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

# daten
L = np.linspace(0,6)
r_1 = 1.4
r_2 = 1.4

# gemessene intensitaeten
I = np.genfromtxt("messwerte_e.txt", unpack=True)
I_polar = np.genfromtxt("polarisation.txt", unpack=True)
phi_plot = np.linspace(0, 360, 37)
I_mode01 = np.genfromtxt("mode01.txt", unpack=True)

# beugungssachenD
lschirm = 142 # achtung this is cm !!
banderror = 0.2
g80  = unp.uarray([14.5, 29.0, 43.5, 58.5, 82.5, 99.0], np.ones(6) * banderror) # aufsteigend
g100 = unp.uarray([18.5, 37.0, 55.5, 86.0, 96.5, 2 * 59.5], np.ones(6) * banderror)
g80  *= 0.5
g100 *= 0.5

# functions
def wavelen(gconst, dist, dn):
    arr = np.array(dn)
    for n in range(1, 1 + len(dn)):
        arr[n - 1] = gconst * unp.sin(unp.arctan(dn[n - 1] / dist)) / n
    return(arr)

# rechungen

# wellenlaenge aus gitter
lambda80  = wavelen(0.001 / 80, lschirm, g80)
lambda100 = wavelen(0.001 / 100, lschirm, g100)

print("\nGITTER 80: ", lambda80)
print("GITTER 80 MEAN: ", np.mean(lambda80))
print("\nGITTER 100: ", lambda100)
print("GITTER 100 MEAN: ", np.mean(lambda100))

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


d_plot = np.linspace(-20, 30, 51)
plt.plot(d_plot, I, 'x')
plt.savefig('build/intensity.pdf')
plt.close()


plt.plot(phi_plot, I_polar, 'x')
plt.savefig('build/polarisation.pdf')
plt.close()


d_plot_mode01 = np.linspace(-20, 10, 31)
plt.plot(d_plot_mode01, I_mode01, 'x')
plt.savefig('build/mode01.pdf')
plt.close()
