import matplotlib.pyplot as plt
import numpy as np
# from tabelle import tabelle
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy import optimize as opt

# daten
# gemessene intensitaeten
THETA_det_scan, Int_det_scan   = np.genfromtxt("messwerte/det_scan.txt", unpack=True)
theta_untergrund , int_untergrund = np.genfromtxt("messwerte/mess1_untergrund.txt", unpack=True)
theta_messung , int_messung = np.genfromtxt("messwerte/mess1.txt", unpack=True)
Theta_rock , Int_rock = np.genfromtxt("messwerte/rock_scan_0.txt", unpack=True)

#Winkel < 0.25 werden vernachlässigt
#  da der Strahl noch teilweise
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

def Gaus(x,a,mu,sigma):
    return a * np.exp(-(x-mu)**2 /(2 * sigma**2) )

def Betragsfunktion(x,a,b,c):
    return a * np.abs(x+b) + c


# Brechungsindex
n_1 = 1 #Luft
n_2 = 1 - 1e-6
#Rauigkeit

sigma_1 = 8e-10 #Schicht
sigma_2 = 3e-10 #Substrat
n_3 = 1 - 2e-6
#Schichtdicke

z2 = 500e-10

def kann_alles_macht_alles(Winkel,n1,n2,n3,sigma1,sigma2,z2):
    #Einfallswinkel
    ai = np.array(Winkel * np.pi / 180 )

    # Wellenvektorübertrag

    qz = 4 * np.pi / 1.54 * np.sin(ai)

    #Betrag des Wellenvektors

    k = 2 * np.pi / 1.54 * 1e10
    # z-Komponenten

    kz1 = k * np.sqrt(n1**2 - np.cos(ai)**2)
    kz2 = k * np.sqrt(n2**2 - np.cos(ai)**2)
    kz3 = k * np.sqrt(n3**2 - np.cos(ai)**2)


    #z-Komponenten

    r12 = (kz1 - kz2) / (kz1 + kz2) * np.exp(-2 *  kz1 * kz2 * sigma1**2)
    r23 = (kz2 - kz3) / (kz2 + kz3) * np.exp(-2 *  kz2 * kz3 * sigma2**2)
    x2 = np.exp(-2j * kz2 * z2) * r23
    x1 = (r12 + x2) / (1 + r12 * x2)
    return x1

params_det, cov_det = curve_fit(Gaus ,THETA_det_scan,Int_det_scan,p0=[0.9e8,0,0.02])
uparams_pol = unp.uarray(params_det, np.sqrt(np.diag(cov_det)))




# fit reflektivitaet gesamt
params_det, cov_det = curve_fit(Gaus ,THETA_det_scan,Int_det_scan,p0=[0.9e8,0,0.02])
uparams_pol = unp.uarray(params_det, np.sqrt(np.diag(cov_det)))

plt.figure(100)
# plt.plot(Theta_messung, np.abs(x1)**2)
plt.plot(Theta_messung, (Int_messung-Int_untergrund)/params_det[0],'-', label='Messung')
plt.yscale('log')
plt.savefig('build/Programm.pdf')




plt.figure(1)
plt.plot(THETA_det_scan, Int_det_scan,'x' ,label='Kurve')
plt.plot(THETA_det_scan, Gaus(THETA_det_scan,*params_det),'-')
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


params_rock, cov_rock = curve_fit(Betragsfunktion ,Theta_rock[(Theta_rock > -0.25) & (Theta_rock < 0.7) ], Int_rock[(Theta_rock > -0.25) & (Theta_rock < 0.7)], p0=[14e7,-0.25,7e7])
uparams_rock = unp.uarray(params_rock, np.sqrt(np.diag(cov_rock)))


# finde nullstelle von rockingscan
#ableitung von betragsfunction
def ableitungbetrag(x):
    return(params_rock[0]  * sign(x) )

def betragsfunktion_params(x):
    return(Betragsfunktion(x, params_rock[0], 0, params_rock[2]))
nullstellen = opt.root(betragsfunktion_params, [1], jac=ableitungbetrag, method='hybr')
print("\n\nNullstellen von Betragsfunktion: ", nullstellen)

plt.figure(4)
plt.plot(Theta_rock, Int_rock,'x', label='Kurve')
plt.plot(Theta_rock, Betragsfunktion(Theta_rock ,*params_rock))
plt.xlabel(r'$\Theta$')
plt.ylabel(r'$Intesity$')
plt.legend(loc='best')
plt.ylim(0,8e7)
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
