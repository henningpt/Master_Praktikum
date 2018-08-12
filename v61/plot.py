import matplotlib.pyplot as plt
import numpy as np
from tabelle import tabelle
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import scipy.constants as const
# daten


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

tabelle(np.array([unp.nominal_values(g80),unp.std_devs(g80),unp.nominal_values(lambda80*1e9), unp.std_devs(lambda80*1e9) ,unp.nominal_values(g100),unp.std_devs(g100),unp.nominal_values(lambda100*1e9),unp.std_devs(lambda100*1e9)]),"wellenlaenge_table",np.array([1,1,1,1,1,1,1,1]))


print("\nGITTER 80: ", lambda80)
print("GITTER 80 unp.mean: ", np.mean(lambda80))
print("\nGITTER 100: ", lambda100)
print("GITTER 100 unp.mean: ", np.mean(lambda100))


#Stabilitätsbedingung
L = np.linspace(0,3)
r_1 = 1.4
r_2 = 1.4
plt.figure(1)
plt.plot(L,(1-L/r_1)*(1-L/r_2) , '-' , label=r'g1g2_r_1=1.4')
r_1=1
plt.plot(L,(1-L/r_1)*(1-L/r_2) , '-' , label=r'g1g2_r_1=1')
plt.plot(L,(1)*(1-L/r_2) , '-' , label=r'g1g2_r_1=999999')
plt.hlines(1,0,4)
plt.hlines(0,0,4)
plt.legend(loc='best')
# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.tight_layout()
plt.savefig('build/plot_lange.pdf')
plt.close()

def g1g2(L):
    r_1=1.4
    r_2=1.4
    return((1-L/r_1)*(1-L/r_2))


r_1 = 1.4
r_2 = 1.4

laenge_laser, leistung_laser = np.genfromtxt("stabil.txt",unpack=True)
laenge_laser*=1e-2
print("lol",g1g2(laenge_laser)[g1g2(laenge_laser)<0])
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(laenge_laser, leistung_laser,'x',label=r'Messwerte')
axarr[0].set_ylabel(r'$P /\si{\milli\watt}$')

axarr[0].legend(loc="upper right")

axarr[1].plot(laenge_laser, g1g2(laenge_laser),'x')
axarr[1].plot(L,(1-L/r_1)*(1-L/r_2) , '-' , label=r'$g1\cdot g2$',alpha=0.5)
axarr[1].plot(L,L*0+1,'k',alpha=0.5)
axarr[1].plot(L,L*0,'k',alpha=0.5)
axarr[1].set_xlabel(r'$L /\si{\meter}$')
axarr[1].set_ylabel(r'$g_1\cdot g_2$')
axarr[1].legend(loc="upper right")
#axarr[0].set_title('Sharing X axis')
f.savefig("build/g1g2_Leistung.pdf")
# plt.figure(2)
# plt.plot(g1g2(laenge_laser), leistung_laser, 'x')
# plt.savefig("build/g1g2_Leistung.pdf")
# plt.close()
tabelle(np.array([laenge_laser,g1g2(laenge_laser),leistung_laser]),"stabil_table",np.array([3,4,2]))

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
plt.savefig('build/mode00.pdf')
plt.close()
print("\n \n Mode00fitparameter:",uparams_mode_00)

tabelle(np.array([d_plot,I]),"mode00_table",np.array([1,1]))



def polarisation(x,I_0,phi_0):
    return I_0 * np.cos(x * np.pi/180 - phi_0)**2
x_pol = np.linspace(0,360)
params_pol, cov_pol =curve_fit(polarisation,phi_plot,I_polar,p0=[142,0])
uparams_pol = unp.uarray(params_pol, np.sqrt(np.diag(cov_pol)))
plt.plot(phi_plot, I_polar, 'x')
plt.plot(x_pol,polarisation(x_pol,*params_pol),'-')
plt.savefig('build/polarisation.pdf')
plt.close()

print("\n \n Polarisationsfitparameter:",uparams_pol)
print("in degree:",uparams_pol[1]*180/np.pi)
tabelle(np.array([phi_plot,I_polar]),"polarisation_table",np.array([0,1]))


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
print("\n \n Mode01fitparameter:",uparams_mode_01)

tabelle(np.array([d_plot_mode01,I_mode01]),"mode01_table",np.array([1,1]))

#longitudiale moden

freq_L1=unp.uarray([214,428,641,855,1069,1283],np.ones(6)*5)*1e6

freq_L2=unp.uarray([188,375,559,746,934,1121,1305],np.ones(7)*5)*1e6

delta_freq_L1= freq_L1[1:]-freq_L1[:-1]
delta_freq_L2= freq_L2[1:]-freq_L2[:-1]

L1 = 0.735
L2 = 0.835

print("\n\nFreq diff für L1",delta_freq_L1)
print("\n\nFreq diff für L2",delta_freq_L2)
# print("\n\nL1=",delta_freq_L1 * 2 / 3e8)
# print("\n\nL2=",delta_freq_L2 * 2 / 3e8)
print("vergleichs wert",const.c)
longi_messung1 = 2 * delta_freq_L1 * L1
longi_messung2 = 2 * delta_freq_L2 * L2
print("longi_messung1:", longi_messung1)
print("longi_messung2:", longi_messung2)

print("\nmittelwert1: ",np.mean(longi_messung1))
print("\nmittelwert2: ",np.mean(longi_messung2))

print("\nrelative abweichung1 : ",(np.mean(longi_messung1)-const.c)/const.c)
print("\nrelative abweichung2 : ",(np.mean(longi_messung2)-const.c)/const.c)

tabelle(np.array([unp.nominal_values(delta_freq_L1)*1e-6,unp.std_devs(delta_freq_L1)*1e-6,unp.nominal_values(longi_messung1)*1e-8,unp.std_devs(longi_messung1)*1e-8]),"freq_L1_table",np.array([0,0,2,2]))
tabelle(np.array([unp.nominal_values(delta_freq_L2)*1e-6,unp.std_devs(delta_freq_L2)*1e-6,unp.nominal_values(longi_messung2)*1e-8,unp.std_devs(longi_messung2)*1e-8]),"freq_L2_table",np.array([0,0,2,2]))
tabelle(np.array([unp.nominal_values(freq_L1)*1e-6,unp.std_devs(freq_L1)*1e-6,unp.nominal_values(freq_L2)[:-1]*1e-6,unp.std_devs(freq_L2)[:-1]*1e-6]),"freq_table",np.array([0,0,0,0]))
