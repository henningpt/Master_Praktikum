import matplotlib.pyplot as plt
import numpy as np
# from tabelle import tabelle
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import scipy.constants as const
# daten
# gemessene intensitaeten
THETA_det_scan, Int_det_scan   = np.genfromtxt("messwerte/det_scan.txt", unpack=True)
theta_untergrund , int_untergrund = np.genfromtxt("messwerte/mess1_untergrund.txt", unpack=True)
theta_messung , int_messung = np.genfromtxt("messwerte/mess1.txt", unpack=True)
Theta_rock , Int_rock = np.genfromtxt("messwerte/rock_scan_0.txt", unpack=True)

#Winkel < 0.25 werden vernachlässigt
# da der Strahl noch teilweise 
# direkt auf den Detektor fällt.

Theta_messung = theta_messung[theta_messung > 0.25]
Int_messung = int_messung[theta_messung > 0.25]
Theta_untergrund  = theta_untergrund[theta_untergrund > 0.25]
Int_untergrund = int_untergrund[theta_untergrund > 0.25]

#Gausfunktion

def Gaus(x,mu,sigma):
    return 1/np.sqrt(2 * np.pi *sigma**2) * np.exp(-(x-mu)**2 /(2 * sigma**2) )

# Brechungsindex
n1 = 1 #Luft
n2 = 1 - 1e-6
n3 = 1 - 2e-6

#Rauigkeit

sigma1 = 8e-10 #Schicht
sigma2 = 3e-10 #Substrat 


#Schichtdicke

z2 = 500e-10

#Einfallswinkel
ai = np.array(Theta_messung * np.pi / 180 )

# Wellenvektorübertrag

qz = 4 * np.pi / 1.54 * np.sin(ai)

#Betrag des Wellenvektors

k = 2 * np.pi / 1.54 * 1e10
# z-Komponenten 

kz1 = k * np.sqrt(n1**2 - np.cos(ai)**2)
kz2 = k * np.sqrt(n2**2 - np.cos(ai)**2)
kz3 = k * np.sqrt(n3**2 - np.cos(ai)**2)


params_det, cov_det = curve_fit(Gaus ,THETA_det_scan,Int_det_scan,)
uparams_pol = unp.uarray(params_pol, np.sqrt(np.diag(cov_pol)))


plt.figure(1)
plt.plot(THETA_det_scan, Int_det_scan, label='Kurve')
plt.xlabel(r'$\Theta$')
plt.ylabel(r'$Intesity$')
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

plt.figure(4)
plt.plot(Theta_rock, Int_rock, label='Kurve')
plt.xlabel(r'$\Theta$')
plt.ylabel(r'$Intesity$')
plt.legend(loc='best')
plt.savefig('build/plot_rocking.pdf')

plt.figure(5)
plt.plot(Theta_messung, Int_messung-Int_untergrund,'-', label='Differenz')
plt.plot(Theta_messung, Int_messung,'--' ,label='Reflektivitätsscan')
plt.plot(Theta_untergrund, Int_untergrund,'-', label='Diffuser Scan')
plt.xlabel(r'$\Theta$')
plt.ylabel(r'$Intesity$')
plt.yscale('log')
plt.legend(loc='best')
plt.savefig('build/plot_messung_untergrund.pdf')



