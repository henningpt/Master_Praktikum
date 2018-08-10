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

#00 mode intensity

def mode00(x,I_0,w,d_0):
    return I_0 * np.exp(-2*((x-d_0)/w)**2)

d_plot = np.linspace(-20, 30, 51)
x_mode_00 = np.linspace(-20,30)
params_mode_00, cov_mode_00 =curve_fit(mode00,d_plot,I,p0=[60,1,15])
uparams_mode_00 = unp.uarray(params_mode_00, np.sqrt(np.diag(cov_mode_00)))
plt.plot(d_plot, I, 'x')
plt.plot(x_mode_00,mode00(x_mode_00,*params_mode_00),'-')
#plt.plot(x_mode_00,mode00(x_mode_00,60,1,15),'-',label=r"test")
plt.legend(loc='best')
# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.tight_layout()
plt.savefig('build/intensity.pdf')
plt.close()


def polarisation(x,I_0,phi_0):
    return I_0 * np.cos(x * np.pi/180 - phi_0)**2
x_pol = np.linspace(0,360)
params_pol, cov_pol =curve_fit(polarisation,phi_plot,I_polar)
uparams_pol = unp.uarray(params_pol, np.sqrt(np.diag(cov_pol)))
plt.plot(phi_plot, I_polar, 'x')
plt.plot(x_pol,polarisation(x_pol,*params_pol),'-')
plt.savefig('build/polarisation.pdf')
plt.close()



def mode01_asym(x,I_01,I_02,w1,w2,d_01,d_02):
    return I_01 * np.exp(-2*((x-d_01)/w1)**2) + I_02 * np.exp(-2*((x-d_02)/w2)**2)




# plots:
d_plot_mode01 = np.linspace(-20, 10, 31)
x_mode_01 = np.linspace(-20,10,1000)
params_mode_01, cov_mode_01 =curve_fit(mode01_asym,d_plot_mode01,I_mode01,p0=[800,200,1,1,-10,0])
uparams_mode_01 = unp.uarray(params_mode_01, np.sqrt(np.diag(cov_mode_01)))
plt.plot(d_plot_mode01, I_mode01, 'x')
plt.plot(x_mode_01, mode01_asym(x_mode_01,*params_mode_01),'r')
plt.savefig('build/mode01.pdf')
plt.close()


#longitudiale moden

freq_L1=unp.uarray([214,428,641,855,1069,1283],np.ones(6)*5)*1e6

freq_L2=unp.uarray([188,375,559,746,934,1121,1305],np.ones(7)*5)*1e6

delta_freq_L1= freq_L1[1:]-freq_L1[:-1]
delta_freq_L2= freq_L2[1:]-freq_L2[:-1]

print("\n\nFreq diff für L1",delta_freq_L1)
print("\n\nFreq diff für L2",delta_freq_L2)

print("\n\nL1=",delta_freq_L1 * 2 / 3e8)
print("\n\nL2=",delta_freq_L2 * 2 / 3e8)
