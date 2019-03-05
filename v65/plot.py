import matplotlib.pyplot as plt
import numpy as np
# from tabelle import tabelle
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy import optimize as opt
from scipy.signal import argrelmin as rel_min
# daten
# gemessene intensitaeten
THETA_det_scan, Int_det_scan   = np.genfromtxt("messwerte/det_scan.txt", unpack=True)
theta_untergrund , int_untergrund = np.genfromtxt("messwerte/mess1_untergrund.txt", unpack=True)
theta_messung , int_messung = np.genfromtxt("messwerte/mess1.txt", unpack=True)
Theta_rock , Int_rock = np.genfromtxt("messwerte/rock_scan_0.txt", unpack=True)

# Winkel < 0.25 werden vernachlässigt
# da der Strahl noch teilweise
# direkt auf den Detektor fällt.

Theta_messung = theta_messung[theta_messung > 0.25]
Int_messung = int_messung[theta_messung > 0.25]
Theta_untergrund  = theta_untergrund[theta_untergrund > 0.25]
Int_untergrund = int_untergrund[theta_untergrund > 0.25]



# signum function
def sign(x):
    if(x >= 0):
        return(x)
    else:
        return(-x)


#Gausfunktion

def Gaus(x,a,mu,w):
    return a * np.exp(-(x-mu)**2 /(w**2) )


def Betragsfunktion(x,a,b,c):
    return a * np.abs(x+b) + c

# Wellenlänge

lam = 1.54e-10


# alpha_g
# def geo(alpha_g, D, d0):
#     return(unp.sin(alpha_g))
# Brechungsindex
n_1 = 1 #Luft
n_2 = 1 - 1e-6
#Rauigkeit

sigma_1 = 10e-10 #Schicht
sigma_2 = 1e-10 #Substrat
n_3 = 1 - 2e-6
#Schichtdicke

z_2 = 500e-10


def kann_alles_macht_alles(Winkel,sigma1,sigma2,z2):
    #Einfallswinkel
    ai = np.array(Winkel * np.pi / 180 )

    # Wellenvektorübertrag

    qz = 4 * np.pi / 1.54 * 1e10 * np.sin(ai)

    #Betrag des Wellenvektors

    k = 2 * np.pi / 1.54 * 1e10
    # z-Komponenten

    kz1 = k * np.sqrt(n_1**2 - np.cos(ai)**2)
    kz2 = k * np.sqrt(n_2**2 - np.cos(ai)**2)
    kz3 = k * np.sqrt(n_3**2 - np.cos(ai)**2)


    #z-Komponenten

    r12 = (kz1 - kz2) / (kz1 + kz2) * np.exp(-2 *  kz1 * kz2 * sigma1**2)
    r23 = (kz2 - kz3) / (kz2 + kz3) * np.exp(-2 *  kz2 * kz3 * sigma2**2)
    x2 = np.exp(-2j * kz2 * z2) * r23
    x1 = (r12 + x2) / (1 + r12 * x2)
    return np.abs(x1)**2

params_det, cov_det = curve_fit(Gaus ,THETA_det_scan,Int_det_scan,p0=[0.9e8,0,0.02])
uparams_det = unp.uarray(params_det, np.sqrt(np.diag(cov_det)))

detektor_radius = 0.5 # Schätzwert!!!!!!

strahl_durchmesser =  0.5 * 2 * unp.sin(uparams_det[2]*np.pi/180) # noch nicht richtig !!!!!
print("strahl_durchmesser=",strahl_durchmesser)

params_rock, cov_rock = curve_fit(Betragsfunktion ,Theta_rock[(Theta_rock > -0.25) & (Theta_rock < 0.7) ],
                                    Int_rock[(Theta_rock > -0.25) & (Theta_rock < 0.7)], p0=[14e7,-0.25,7e7])
uparams_rock = unp.uarray(params_rock, np.sqrt(np.diag(cov_rock)))


# finde nullstelle von rockingscan
def betragsfunktion_params(x):
    return(Betragsfunktion(x, params_rock[0], 0, params_rock[2]))

Winkel_rock = - uparams_rock[2]/uparams_rock[0]
print("\n\nGeometry winkel ",Winkel_rock)
Winkel_rock_rad = Winkel_rock * np.pi / 180

print("\n Winkel in rad ", Winkel_rock_rad)
proben_durchmesser = strahl_durchmesser / unp.sin(Winkel_rock_rad)

print("proben_durchmesser", proben_durchmesser)

geo_winkel = np.arcsin(noms(strahl_durchmesser)/noms(proben_durchmesser))*180/np.pi
print("Anderer Geometriewinkel???",geo_winkel)

plt.figure(4)
plt.plot(Theta_rock, Int_rock,'x', label='Kurve')
plt.plot(Theta_rock, Betragsfunktion(Theta_rock ,*params_rock))
plt.xlabel(r'$\Theta$')
plt.ylabel(r'$Intesity$')
plt.legend(loc='best')
plt.ylim(0,8e7)
plt.savefig('build/plot_rocking.pdf')

Int_messung_sauber = (Int_messung - Int_untergrund) / params_det[0]  # Int_messung_sauber ist auf I0 normiert -> reflektivitaet
Int_messung_sauber_mit_geo_faktor = np.array(Int_messung_sauber)
Int_messung_sauber_mit_geo_faktor[Theta_messung < geo_winkel] = Int_messung_sauber[Theta_messung < geo_winkel] / ( noms(proben_durchmesser) * np.sin(Theta_messung[Theta_messung < geo_winkel]*np.pi/180) / noms(strahl_durchmesser))
print(Int_messung_sauber_mit_geo_faktor[Theta_messung < Winkel_rock])
print("Geometriefaktor",noms(proben_durchmesser) * np.sin(Theta_messung[Theta_messung < Winkel_rock]*np.pi/180) / noms(strahl_durchmesser))
print(Int_messung_sauber_mit_geo_faktor- Int_messung_sauber)
# fit reflektivitaet gesamt
Theta_messung_ohne = Theta_messung[Theta_messung > Winkel_rock]
Int_messung_sauber_ohne = Int_messung_sauber[Theta_messung > Winkel_rock]

index_min = rel_min(np.log(Int_messung_sauber),order=4)
alpha_min = Theta_messung[index_min]
delta_a_i = alpha_min[:-1] -alpha_min[1:]
print("\nDelta_a_i",delta_a_i)
delta_a_i_mean = np.mean(delta_a_i)
delta_a_i_mean_in_rad = delta_a_i_mean * np.pi / 180
#Berechnete Schichtdicke
z_berechnet = lam / (2 *delta_a_i_mean_in_rad)

print("Schichtdicke", z_berechnet)

# params_messung, cov_messung = curve_fit(kann_alles_macht_alles, Theta_messung_ohne,
#                                         Int_messung_sauber_ohne, p0=[sigma_1, sigma_2, z_berechnet])
# uparams_messung = unp.uarray(params_messung, np.sqrt(np.diag(cov_messung)))



sigma_1 = 9e-10 #Schicht
sigma_2 = 4.5e-10 #Substrat
#
# test_1 = 9e-10 #Schicht
# test_2 = 4.5e-10 #Substrat
#
# test_3 = 10e-10 #Schicht
# test_4 = 4.5e-10 #Substrat

# Beobachtung:  sigma substrat bestimmt die mittlere Steigung am Ende
#               sigma Schicht bestimmt die Stärke der Schwingungen aber antiproportional
plt.figure(100)
# plt.plot(Theta_messung, kann_alles_macht_alles(Theta_messung,*params_messung),label='Fit')
plt.plot(Theta_messung, Int_messung_sauber_mit_geo_faktor,'-', label='Messung_mit_Faktor')
plt.plot(Theta_messung, Int_messung_sauber,'-', label='Messung')

# plt.plot(Theta_messung, kann_alles_macht_alles(Theta_messung, sigma_1, sigma_2, z_berechnet)*10**2,label='Hand Fit')
# plt.plot(Theta_messung, kann_alles_macht_alles(Theta_messung, test_1, test_2, z_berechnet)*10**2,label='Hand Fit')
# plt.plot(Theta_messung, kann_alles_macht_alles(Theta_messung, test_3, test_4, z_berechnet)*10**2,label='Hand Fit ')
plt.plot(Theta_messung[index_min],Int_messung_sauber[index_min],'x')
plt.yscale('log')
plt.legend(loc="best")
plt.savefig('build/Programm.pdf')


plt.figure(101)
plt.plot(Theta_messung, Int_messung_sauber_mit_geo_faktor,'-', label='$I_{korr}$')
plt.plot(Theta_messung, Int_messung_sauber,'--', label='$I_{mess}$')
plt.plot(Theta_messung[index_min],Int_messung_sauber_mit_geo_faktor[index_min],'x',label=r'$\alpha_i$ Minima')
plt.yscale('log')
plt.xlabel(r'$\alpha_i / °$')
plt.ylabel(r'Intensität')
plt.legend(loc="best")
plt.savefig('build/geometriefaktor.pdf')




plt.figure(1)
plt.plot(THETA_det_scan, Int_det_scan,'x' ,label='Kurve')
plt.plot(THETA_det_scan, Gaus(THETA_det_scan,*params_det),'-')
plt.xlabel(r'$\Theta$')
plt.ylabel(r'$Intesität$')
plt.legend(loc='best')
plt.savefig('build/plot_det_scan.pdf')



plt.figure(2)
plt.plot(Theta_untergrund, Int_untergrund, label='Kurve')
plt.xlabel(r'$\Theta$')
plt.ylabel(r'$Intesity$')
plt.legend(loc='best')
plt.savefig('build/plot_untergrund.pdf')

plt.figure(3)
plt.plot(Theta_messung, Int_messung,'x' ,label='Kurve')
plt.xlabel(r'$\Theta$')
plt.ylabel(r'$Intesity$')
plt.legend(loc='best')
plt.savefig('build/plot_messung.pdf')


plt.figure(5)
plt.plot(Theta_messung, Int_messung-Int_untergrund,'-', label='Differenz')
plt.plot(Theta_messung, Int_messung,'--' ,label='Reflektivitätsscan')
plt.plot(Theta_untergrund, Int_untergrund,'-', label='Diffuser Scan')
plt.xlabel(r'$\Theta$')
plt.ylabel(r'$Intesity$')
plt.yscale('log')
plt.legend(loc='best')
plt.savefig('build/plot_messung_untergrund.pdf')
