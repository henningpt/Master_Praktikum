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

Theta_messung = theta_messung[theta_messung > 0.05]
Int_messung = int_messung[theta_messung > 0.05]
Theta_untergrund  = theta_untergrund[theta_untergrund > 0.05]
Int_untergrund = int_untergrund[theta_untergrund > 0.05]




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
n_2 = 1 - 3.5e-6
n_3 = 1 - 7.9e-6
#Rauigkeit

sigma_1 = 10e-10 #Schicht
sigma_2 = 1e-10 #Substrat
#Schichtdicke

z_2 = 500e-10


def kann_alles_macht_alles(Winkel,sigma1,sigma2,z2,n2,n3):
    #Einfallswinkel
    ai = np.array(Winkel * np.pi / 180 )
    if(n2>1 or n3>1 ):
        return 100000000
    if(sigma1<0 or sigma2<0 ):
        return 100000000

    # Wellenvektorübertrag

    qz = 4 * np.pi / 1.54 * 1e10 * np.sin(ai)

    #Betrag des Wellenvektors

    k = 2 * np.pi / 1.54 * 1e10
    # z-Komponenten
    kz1 = k * np.sqrt(n_1**2 - np.cos(ai)**2 + 0j)
    kz2 = k * np.sqrt(n2**2 - np.cos(ai)**2  + 0j)
    kz3 = k * np.sqrt(n3**2 - np.cos(ai)**2  + 0j)


    #z-Komponenten

    r12 = (kz1 - kz2) / (kz1 + kz2) * np.exp(-2 *  kz1 * kz2 * sigma1**2)
    r23 = (kz2 - kz3) / (kz2 + kz3) * np.exp(-2 *  kz2 * kz3 * sigma2**2)
    x2 = np.exp(- 1j *2 * kz2 * z2) * r23
    # print("test",(r12 + x2),"durch", (1 + r12 * x2))
    x1 = (r12 + x2) / (1 + r12 * x2)
    antwort =  np.log(np.abs(x1)**2)
    return antwort



params_det, cov_det = curve_fit(Gaus ,THETA_det_scan,Int_det_scan,p0=[0.9e8,0,0.02])
uparams_det = unp.uarray(params_det, np.sqrt(np.diag(cov_det)))

detektor_radius = 100 # Schätzwert!!!!!!

strahl_durchmesser =  detektor_radius * 2 * unp.sin(uparams_det[2]*np.pi/180) # noch nicht richtig !!!!!
print("strahl_durchmesser=",strahl_durchmesser)
print("Intesität=",uparams_det[0])
params_rock, cov_rock = curve_fit(Betragsfunktion ,Theta_rock[(Theta_rock > -0.25) & (Theta_rock < 0.7) ],
                                    Int_rock[(Theta_rock > -0.25) & (Theta_rock < 0.7)], p0=[14e7,-0.25,7e7])
uparams_rock = unp.uarray(params_rock, np.sqrt(np.diag(cov_rock)))
print("Rocking_scan_params",uparams_rock)

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
# print("Anderer Geometriewinkel???",geo_winkel)

plt.figure(4)
plt.plot(Theta_rock, Int_rock,'x', label='Kurve')
plt.plot(Theta_rock, Betragsfunktion(Theta_rock ,*params_rock))
plt.xlabel(r'$\Theta/°$')
plt.ylabel(r'$Intesity I$')
plt.legend(loc='best')
plt.ylim(0,8e7)
plt.savefig('build/plot_rocking.pdf')

Int_messung_sauber = (Int_messung - Int_untergrund) # / params_det[0]  # Int_messung_sauber ist auf I0 normiert -> reflektivitaet
Int_messung_sauber_mit_geo_faktor = np.array(Int_messung_sauber)
Int_messung_sauber_mit_geo_faktor[Theta_messung < geo_winkel] = Int_messung_sauber[Theta_messung < geo_winkel] / ( noms(proben_durchmesser) * np.sin(Theta_messung[Theta_messung < geo_winkel]*np.pi/180) / noms(strahl_durchmesser))
print(Int_messung_sauber_mit_geo_faktor[Theta_messung < Winkel_rock])
# print("Geometriefaktor",noms(proben_durchmesser) * np.sin(Theta_messung[Theta_messung < Winkel_rock]*np.pi/180) / noms(strahl_durchmesser))
# print(Int_messung_sauber_mit_geo_faktor- Int_messung_sauber)
Int_messung_sauber_mit_geo_faktor = Int_messung_sauber_mit_geo_faktor/Int_messung_sauber_mit_geo_faktor[0]

plt.figure(44)
plt.plot(theta_messung,(int_messung-int_untergrund) / params_det[0] ,'-', label='Messwerte')
plt.plot(Theta_messung,Int_messung_sauber_mit_geo_faktor ,'-', label="Messwerte mit Geo")
plt.axvline(x=0.05)
plt.xlabel(r'$\Theta/°$')
plt.ylabel(r'$Intesity I$')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('build/plot_Messwerte.pdf')


# fit reflektivitaet gesamt

index_min = rel_min(np.log(Int_messung_sauber),order=4)
alpha_min = Theta_messung[index_min]
alpha_min = alpha_min[alpha_min<2]

delta_a_i = alpha_min[1:] - alpha_min[:-1]
delta_a_i_mean = np.mean(delta_a_i)
delta_a_i_std = np.std(delta_a_i)

# print("\nDelta_a_i_mean",delta_a_i_mean)
# print("\nDelta_a_i_std",delta_a_i_std)
udelta_a_i = unp.uarray(delta_a_i_mean,delta_a_i_std)
delta_a_i_mean_in_rad = udelta_a_i * np.pi / 180
#Berechnete Schichtdicke
z_berechnet = lam / (2 *delta_a_i_mean_in_rad)

print("Schichtdicke", z_berechnet)



# Beobachtung:  sigma substrat bestimmt die mittlere Steigung am Ende
#               sigma Schicht bestimmt die Stärke der Schwingungen aber antiproportional

sigma_1 =   10e-10 #Schicht
sigma_2 =  10e-10 #Substrat
#
# test_1 = 9e-10 #Schicht
# test_2 = 4.5e-10 #Substrat
#
# test_3 = 10e-10 #Schicht
# test_4 = 4.5e-10 #Substrat

n_2 = 1 - 3e-6
n_3 = 1 - 7e-6

n_4 = 1 - 1e-6
n_5 = 1 - 1e-6

# Beobachtung: n3 substrat bestimmt die höhe der funktion
#              n2 Schicht bestimmt die Stärke der Schwingungen protional


# n_6 = 1 + 0.5e-6
# n_7 = 1 - 12e-6

def z_gesucht(Winkel,z):
    sigma_1 = 5e-10 #Schicht
    sigma_2 = 4e-10 #Substrat
    n_2 = 1 - 1.5e-6
    n_3 = 1 - 6e-6
    return(kann_alles_macht_alles(Winkel,sigma_1,sigma_2,z,n_2,n_3))

def n_3_gesucht(Winkel,n3,n2):
    sigma_1 = 5e-10 #Schicht
    sigma_2 = 4e-10 #Substrat
    # n_2 = 1 - 1.5e-6
    return(kann_alles_macht_alles(Winkel,sigma_1,sigma_2,noms(z_berechnet),n_2,n3))

def sigma_gesucht(Winkel,sigma1,sigma2,z):
    n_4 = 1 - 3.5e-6
    n_5 = 1 - 7.6e-6
    return(kann_alles_macht_alles(Winkel,sigma1,sigma2,z,n_4,n_5))


bereich = 150

Testfunktionswerte = kann_alles_macht_alles(Theta_messung[:-bereich],sigma_1, sigma_2, noms(z_berechnet),n_2, n_3)

print("Funkionswerte ",Testfunktionswerte)

params_messung, cov_messung = curve_fit(kann_alles_macht_alles, Theta_messung[:-bereich],
                                        np.log(Int_messung_sauber_mit_geo_faktor[:-bereich]), p0=[
                                         sigma_1, sigma_2, noms(z_berechnet),n_2, n_3
                                        ],maxfev=100000)


uparams_messung = unp.uarray(params_messung, np.sqrt(np.diag(cov_messung)))
print("Params Fit nach 1",uparams_messung)
print("Params von delta Fit nach 1",1-uparams_messung)
n_schicht = uparams_messung[3]
n_substrat = uparams_messung[4]

# params_messung_z, cov_messung_z = curve_fit(z_gesucht, Theta_messung[10:],
#                                           Int_messung_sauber_mit_geo_faktor[10:], p0=[noms(z_berechnet)],maxfev=800)
# uparams_messung_z = unp.uarray(params_messung_z, np.sqrt(np.diag(cov_messung_z)))
#
# params_messung_n3, cov_messung_n3 = curve_fit(n_3_gesucht, Theta_messung[20:],
#                                           Int_messung_sauber_mit_geo_faktor[20:], p0=[n_2,n_3],maxfev=800)
# uparams_messung_n3 = unp.uarray(params_messung_n3, np.sqrt(np.diag(cov_messung_n3)))
# print(uparams_messung_n3)
#
params_messung_sigma, cov_messung_sigma = curve_fit(sigma_gesucht, Theta_messung[4:-bereich],
                                           np.log(Int_messung_sauber_mit_geo_faktor[4:-bereich]), p0=[sigma_1,sigma_2,noms(z_berechnet)],maxfev=800000)
uparams_messung_sigma = unp.uarray(params_messung_sigma, np.sqrt(np.diag(cov_messung_sigma)))
# print(uparams_messung_sigma)

    # n_2 = 1 - 1.5e-6
    # n_3 = 1 - 6e-6

params_test=np.array(params_messung)
# params_test[0] = params_messung[0]-0.9e-9
# params_test[1] = params_messung[1]+0.3e-10
# params_test[3] = 1 - 2.5e-6
# params_test[4] = 1 - 7e-6
params_test[0] = 8e-10+1e-10
params_test[1] = 2e-10+2.9e-10
params_test[3] = 1- 2.1e-6
params_test[4] = 1 - 6e-6 -0.5e-6


#Schicht
# params_test[]=

params_messung, cov_messung = curve_fit(kann_alles_macht_alles, Theta_messung[:-bereich],
                                        np.log(Int_messung_sauber_mit_geo_faktor[:-bereich]), p0=[
                                        *params_test
                                        ])


uparams_messung = unp.uarray(params_messung, np.sqrt(np.diag(cov_messung)))
print("Params Fit nach 2",uparams_messung)
print("Params von delta Fit nach 2",1-uparams_messung)
print("Parmas diff", params_messung-params_test)

n_schicht = uparams_messung[3]
n_substrat = uparams_messung[4]

def sigma_mit_n_fit(Winkel,sigma1,sigma2,z):
     n_4 = noms(n_schicht)
     n_5 = noms(n_substrat)
     return(kann_alles_macht_alles(Winkel,sigma1,sigma2,z,n_4,n_5))

params_messung_sigma_mit_n, cov_messung_sigma_mit_n = curve_fit(sigma_mit_n_fit, Theta_messung[4:-bereich],
                                           np.log(Int_messung_sauber_mit_geo_faktor[4:-bereich]), p0=[sigma_1,sigma_2,noms(z_berechnet)],maxfev=800000)
uparams_messung_sigma_mit_n = unp.uarray(params_messung_sigma_mit_n, np.sqrt(np.diag(cov_messung_sigma_mit_n)))
# print("super test",uparams_messung_sigma_mit_n)




lit_delta_schicht  = 3.5e-6
lit_delta_substrat = 7.6e-6

print("\nDelta schicht lit abweichung: ",
        abs( 1 - n_schicht - lit_delta_schicht) / lit_delta_schicht)
print("Delta substrat lit abweichung: ",
        abs(1 - n_substrat - lit_delta_substrat) / lit_delta_substrat)

def e_dichte(n):
    lam = 1.54e-10
    r_e = 2.8179403227e-15
    return 2*(1-n)*np.pi/(lam**2*r_e)


# lit werte Elektronendichten
c_e_radius_info = const.physical_constants['classical electron radius']
c_e_radius      = unp.uarray(c_e_radius_info[0], c_e_radius_info[2])

lit_edichte_schicht  =  9.5 * 10**(14) / c_e_radius


print("\nlit dichte schicht: ", lit_edichte_schicht)

lit_edichte_substrat = 20.0 * 10**(14) / c_e_radius
print("litdichte substrat: ", lit_edichte_substrat)


print('\nElektronendichten n_schicht', e_dichte(n_schicht))
print('\nElektronendichten n_schicht lit', lit_edichte_schicht)
print("relat. abweichung zu lit wert schicht : ",
        abs(e_dichte(n_schicht) - lit_edichte_schicht) / lit_edichte_schicht)

print('\nElektronendichten n_substrat', e_dichte(n_substrat))
print('\nElektronendichten n_substrat lit', lit_edichte_substrat)
print("relat. abweichung zu lit wert substrat : ",
        abs(e_dichte(n_substrat) - lit_edichte_substrat) / lit_edichte_substrat)

print('\n delta n_schicht',1-n_schicht)
print('delta n_substrat',1-n_substrat)
# print("z_überfit",uparams_messung_z)
# print("n_2,n_3", uparams_messung_n3)
plt.figure(100)
# plt.plot(Theta_messung, kann_alles_macht_alles(Theta_messung,*params_messung),label='Fit')
plt.plot(Theta_messung, Int_messung_sauber_mit_geo_faktor,'tab:orange', label=r'$I_{korr}$')
plt.plot(Theta_messung[:-bereich], Int_messung_sauber_mit_geo_faktor[:-bereich],'--r', label=r'$I_{korr}$ für Fit')
# plt.plot(Theta_messung, kann_alles_macht_alles(Theta_messung, sigma_1, sigma_2, 7e-8,n_2,n_3),label='Hand Fit 3')
# plt.plot(Theta_messung, kann_alles_macht_alles(Theta_messung, sigma_1, sigma_2, noms(z_berechnet),n_2,n_3),label='Hand Fit 1')
# plt.plot(Theta_messung, kann_alles_macht_alles(Theta_messung, sigma_1, sigma_2, noms(z_berechnet),n_6,n_7),label='Hand Fit 7')
# plt.plot(Theta_messung, kann_alles_macht_alles(Theta_messung, sigma_1, sigma_2, z_2,n_4,n_5),label='Hand Fit z1')
plt.plot(Theta_messung, np.exp(kann_alles_macht_alles(Theta_messung, *params_messung)),'tab:blue', linewidth=0.75 ,label=r'Fit ')
# plt.plot(Theta_messung, np.exp(kann_alles_macht_alles(Theta_messung, 0.2e-10, 6e-10, noms(z_berechnet), 1-0.1e-6, 1-2e-6 )) , linewidth=0.75 ,label=r'Fit hand 0 ')
# plt.plot(Theta_messung, np.exp(kann_alles_macht_alles(Theta_messung, 0.2e-10, 6e-10, noms(z_berechnet), 1-0.2e-6, 1-3e-6 )) , linewidth=0.75 ,label=r'Fit hand 1 ')
# plt.plot(Theta_messung, np.exp(kann_alles_macht_alles(Theta_messung, 0.2e-10, 6e-10, noms(z_berechnet), 1-0.3e-6, 1-4e-6 )) , linewidth=0.75 ,label=r'Fit hand 2')
# plt.plot(Theta_messung, np.exp(kann_alles_macht_alles(Theta_messung, 0.2e-10, 6e-10, noms(z_berechnet), 1-0.4e-6, 1-8e-6 )) , linewidth=0.75 ,label=r'Fit hand 3 ')
# plt.plot(Theta_messung, np.exp(kann_alles_macht_alles(Theta_messung, *params_test)),'m' ,linewidth=0.75 ,label='Fit test')
#plt.plot(Theta_messung, z_gesucht(Theta_messung, *params_messung_z),label='z_gesucht')
# plt.plot(Theta_messung, n_3_gesucht(Theta_messung, *params_messung_n3),label='n3_gesucht')
plt.yscale('log')
plt.xlabel(r'$\alpha_i / °$')
plt.ylabel(r'$\ln I$')
plt.legend(loc="best")
plt.savefig('build/Programm.pdf')

plt.figure(1000)
# plt.plot(Theta_messung, kann_alles_macht_alles(Theta_messung,*params_messung),label='Fit')
plt.plot(Theta_messung, Int_messung_sauber_mit_geo_faktor,'tab:orange', label=r'$I_{korr}$')
plt.plot(Theta_messung[4:-bereich], Int_messung_sauber_mit_geo_faktor[4:-bereich],'--r', label=r'$I_{korr}$ für Fit')
plt.plot(Theta_messung, np.exp(kann_alles_macht_alles(Theta_messung, *params_messung)),'tab:blue', linewidth=0.75 ,label=r'Fit ')
plt.plot(Theta_messung, np.exp(sigma_gesucht(Theta_messung, *params_messung_sigma)),label=r'Lit Plot')
plt.plot(Theta_messung, np.exp(sigma_mit_n_fit(Theta_messung, *params_messung_sigma_mit_n)),label=r'fancy Plot')
plt.yscale('log')
plt.xlabel(r'$\alpha_i / °$')
plt.ylabel(r'$\ln I$')
plt.legend(loc="best")
plt.savefig('build/Programm_mit_lit.pdf')



plt.figure(101)
plt.plot(Theta_messung, Int_messung_sauber_mit_geo_faktor,'-', label='$I_{korr}$')
plt.plot(Theta_messung, Int_messung_sauber,'--', label='$I_{mess}$')
plt.plot(Theta_messung[index_min],Int_messung_sauber_mit_geo_faktor[index_min],'x',label=r'$\alpha_i$ Minima')
plt.yscale('log')
plt.xlabel(r'$\alpha_i / °$')
plt.ylabel(r'$\ln I$')
plt.legend(loc="best")
plt.savefig('build/geometriefaktor.pdf')




plt.figure(1)
plt.plot(THETA_det_scan, Int_det_scan,'x' ,label='Kurve')
plt.plot(THETA_det_scan, Gaus(THETA_det_scan,*params_det),'-')
plt.xlabel(r'$\Theta/°$')
plt.ylabel(r'Intensität $I$')
plt.legend(loc='best')
plt.savefig('build/plot_det_scan.pdf')



plt.figure(2)
plt.plot(Theta_untergrund, Int_untergrund, label='Kurve')
plt.xlabel(r'$\Theta/°$')
plt.ylabel(r'Intensität $I$')
plt.legend(loc='best')
plt.savefig('build/plot_untergrund.pdf')

plt.figure(3)
plt.plot(Theta_messung, Int_messung,'x' ,label='Kurve')
plt.xlabel(r'$\Theta/°$')
plt.ylabel(r'Intensität $I$')
plt.legend(loc='best')
plt.savefig('build/plot_messung.pdf')


plt.figure(5)
plt.plot(Theta_messung, Int_messung-Int_untergrund,'-', label='Differenz')
plt.plot(Theta_messung, Int_messung,'--' ,label='Reflektivitätsscan')
plt.plot(Theta_untergrund, Int_untergrund,'-', label='Diffuser Scan')
plt.xlabel(r'$\Theta/°$')
plt.ylabel(r'Intensität $I$')
plt.yscale('log')
plt.legend(loc='best')
plt.savefig('build/plot_messung_untergrund.pdf')
