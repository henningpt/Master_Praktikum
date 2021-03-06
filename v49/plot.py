import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
from scipy.signal import argrelextrema as rextrem
from uncertainties import unumpy as unp
import scipy.constants as con
from tabelle import tabelle
from functions import *

# werte
tau   = 20e-3
apulse = 9.48e-6
bpulse = 3.43e-6
phi   = 111 # degree
f     = 21.71617e6
dprob = 4.4e-3
gyrom = 2.67515255e8 # gyromag protonen in wasser http://kirste.userpage.fu-berlin.de/chemistry/general/constants.html
gyrom_falsch   = 1.0 # nur zum ausprobieren
gyrom_e = 1.7608592e11# gyromag electronen
dichte = 0.9972995 * 1000 # bei 24 grad celsius : https://chemistry.sciences.ncsu.edu/resource/H2Odensity_vp.html

visk_lit = 891e-6
t2_lit = unp.uarray([1.52, 0.093])
t1_lit = unp.uarray([3.09, 0.15])
alpha = 1.024e-9 # apparatur konstante
# lade daten
t1_visk = unp.uarray(12.0 * 60.0 + 3.0 + 0.5, 2)
t2_visk = unp.uarray(12.0 * 60.0 + 0.3, 2)  # zeiten viskosimeter
diff_lit = unp.uarray(2.78e-9, 0.035e-9)

t1_tau, t1_amp = np.genfromtxt("t1messung.txt", unpack=True)
t1_tau *= 1e-3
t1_amp *= 1e-3
d_tau, d_amp = np.genfromtxt("diffmessung.txt", unpack=True)
d_tau *= 2e-3 # umrechnen in sekunden und mal 2 wegen t = 2tau
d_amp *= 1e-3 # umrechenn in volt

t_visk = 0.5 * (t1_visk + t2_visk)
print("visk zeit 1: ", t1_visk)
print("visk zeit 2: ", t2_visk)
print("mittelwert viskosimeter: ", t_visk)

sigma = (-(t_visk- 700)* 0.2 / 100) +  0.9
print("\nsigma: ", sigma)

# lade csv dateien:
mydata = np.genfromtxt('cp.csv', delimiter=',')
time_a = mydata[:,0]
sig_a  = mydata[:,1]

mydata2 = np.genfromtxt('mg.csv', delimiter=',')
time_a2 = mydata2[:,0]
# zeitskala verschieben, sodass keine negativen Zeiten auftreten
time_a2 += abs(min(time_a2))
sig_a2  = mydata2[:,1]


# zum testen zweiter kanal
gruen_t = mydata2[:,0]
gruen_u = mydata2[:,2]

# # functions
# def diffkoeff(t, a, T2, D, g, G):
#     return(a * np.e**(-t / T2) * np.e**(- D * g**2 * G**2 * t**3 / 12.0))

#
# def diffkoeff(t, a, D):
#     return(a * np.e**(-t / t2_a) * np.e**(- D * gyrom**2 * gradient**2 * t**3 / 12.0))


# def diffkoeff(t, a):
#     return(a * np.e**(-t / 1.58) * np.e**(- 5e-10 t**3 / 12.))



# berechne Gradient
print("TEST vorher: ", d_tau, " SIGNAL vorher: ", d_amp)
gradient = grad(dprob, 2.67515255e8, float(d_tau[thalb(d_amp)]))
print("TEST: ", d_tau, " SIGNAL: ", d_amp)
print("GRADIENT: ", gradient)
# verarbeiten
# filtern

sig_a_cut , time_a_cut = filter(sig_a, time_a, 21e-3)
sig_a2_cut = sig_a2
time_a2_cut = time_a2


# peaks suchen

eps = tau
start = time_a2_cut[np.argmax(sig_a2_cut)]
print("START: ", start)
time_a2_cut += -start
extrema_a2 = np.ones(51)
extrema_a2 *= -1
extrema_a2[0] = np.argmax(sig_a2_cut)
for h in range(0, len(extrema_a2) - 1):
    # print("\n\nTESTTEST: ", np.argwhere(abs((time_a2_cut - start) - (h + 1) * 4 * tau) < 0.2e-2)[0])
    #print("\n\nEINTEST: ", (time_a2_cut))
    extrema_a2[h+1] =np.argmax(sig_a2_cut[np.argwhere(abs((time_a2_cut) - (h + 1) * 4 * tau) < eps)]) + np.argwhere(abs((time_a2_cut) - (h + 1) * 4 * tau) < eps)[0]

extrema_a2 = extrema_a2.astype(int)
print("\n\nNEUE EXTREMA: ", extrema_a2)
print("size: ", len(extrema_a2))
# extrema_a = rextrem(sig_a_cut, np.greater, order=4)
# extrema_a2 = rextrem(sig_a2_cut, np.greater, order=29)
np.set_printoptions(precision=10)
print("\nWERTE: ", sig_a2_cut[extrema_a2])
#time_a2 = time_a2[1::2]
#sig_a2  = sig_a2[1::2]

# fits, rechnungen
# a)
# aparams, acov = cf(expf, time_a2_cut[extrema_a2], sig_a2_cut[extrema_a2], maxfev=2000) curve_fit klappt nicht

# berechne viskosität

viskositaet = viskos(dichte, alpha, t_visk , sigma)
print("\n viskosität", viskositaet)
diffkoeff_visk = con.k * (24 + 273.15) /+ (6 * viskositaet *  np.pi * 5e-10)
print("\ndiffkoeff aus visk: ", diffkoeff_visk, "\n")

print("\n\nViskosität rel abweichung: ", abs(viskositaet - visk_lit) / visk_lit)

# log -> linfit
aparams, acov = cf(expf, time_a2_cut[extrema_a2] , sig_a2_cut[extrema_a2], maxfev=10000)
auparams = unp.uarray(aparams, np.sqrt(np.diag(acov)))
# auparams = unp.uarray([1,1,1,1], [1,1,1,1])
t2_a = -1 / auparams[1] # berechne T2 fuer meiboom-gill methode

# function fuer diffusion

def diffkoeff(t, a, D):
    return(a * np.e**(-t / t2_a.nominal_value) * np.e**(- D * (8.8 / (dprob * float(d_tau[thalb(d_amp)])))**2 * t**3 / 12.0))


# fit fuer Diffusionvalue
dparams, dcov = cf(diffkoeff, d_tau, d_amp)
duparams = unp.uarray(dparams, np.sqrt(np.diag(dcov)))

# fit fuer t1
# t1params, t1cov = cf(t1_bestimmung, t1_tau, t1_amp)
# t1uparams = unp.uarray(t1params, np.sqrt(np.diag(t1cov)))

# fit fuer t1 neu
t1params, t1cov = cf(lnr, t1_tau[:-1], np.log(max(t1_amp) - t1_amp[:-1]))
t1uparams = unp.uarray(t1params, np.sqrt(np.diag(t1cov)))


# plotten
# stplot(t1_tau, t1_amp, 0, 't1')
# t1_plot = np.linspace(min(t1_tau), max(t1_tau))
# plt.errorbar(t1_plot, t1_bestimmung(t1_plot, *t1params), label='Fit', fmt='-')
# plt.legend(loc='best')
# plt.savefig("build/t_u_plott1.pdf")
# plt.close()




stplot(gruen_t, gruen_u, 0, 'gruen')
plt.close()



# t1 wie in auswertung beschrieben


stplot(t1_tau[:-1], np.log(max(t1_amp) - t1_amp[:-1]), 1, 't1')
t1_plot = np.linspace(min(t1_tau), max(t1_tau[:-1]))
plt.errorbar(t1_plot, lnr(t1_plot, *t1params), label='Fit', fmt='-')
plt.legend(loc='best')
plt.savefig("build/t_u_plott1.pdf")
plt.close()

stplot(d_tau, d_amp, 0, 'diff')
d_plot = np.linspace(min(d_tau), max(d_tau))
plt.errorbar(d_plot, diffkoeff(d_plot, *dparams), label='Fit', fmt='-')
# plt.plot(d_plot, diffkoeff(d_plot, 0.79, 1e-9))
plt.legend(loc='best')
plt.savefig("build/t_u_plotdiff.pdf")
plt.close()

stplot(time_a2, sig_a2, 0, "2")
plt.plot(time_a2_cut[extrema_a2], sig_a2_cut[extrema_a2], 'x',label='Ausgewählte Messdaten')
plt.legend(loc='best')
plt.savefig('build/t_u_plot2.pdf')
plt.close()

# stplot(time_a2_cut[extrema_a2], sig_a2_cut[extrema_a2], 0, "2_extrem")
# t_plot = np.linspace(min(time_a2_cut[extrema_a2]), max(time_a2_cut[extrema_a2]))
# plt.errorbar(t_plot, expf(t_plot, *aparams), yerr=0, label='Fit', fmt='-')
# plt.savefig("build/t_u_plot" + "2_extrem" + ".pdf")
# plt.close()

#stplot(time_a2_cut, sig_a2_cut, 0, "2_extrem")
t_plot = np.linspace(min(time_a2_cut), max(time_a2_cut) )
plt.plot(t_plot - start, expf(t_plot - start, *aparams),'-', label='Fit')
plt.plot(time_a2_cut[extrema_a2], sig_a2_cut[extrema_a2],  'x',  label ='Echos')
# for k in range(0, len(extrema_a2)):
#      plt.axvline(x=time_a2_cut[extrema_a2[k]])
plt.legend(loc='best')
plt.savefig("build/t_u_plot" + "2_extrem" + ".pdf")
plt.close()

# berechne halbwertszeit aus fit:
t_half_exact = d_plot[thalb(diffkoeff(d_plot, *dparams))]


# berechne molekuelradius
molrad = radius(viskositaet, duparams[1])


# ausgeben
print("\nT2 berechnet: ", t2_a)
print("params aus cf: ", auparams)
print("t_halb" , d_tau[thalb(d_amp)])
print("exaktwert : ", t_half_exact)
print("\n\nparams from diff_fit: ")
for enum in enumerate(duparams):
    print("\n", enum)
print("\n\nparams from t1_fit: ")
for enum in enumerate(t1uparams):
    print("\n", enum)
print("\n\nmolekülradius: ", molrad)




print("\n\nT1 bestimmt: ", -1.0 / t1uparams[0])
print("\n\nt1_taus in t umgerechnet: ", t1_tau)
print("\nd_taus in t umgerechnet: ", d_tau)

#tabellen
#tabelle(datensatz, Name, Rundungen):  # i=Spalten j=Zeilen
tabelle(np.array([d_tau*1000,d_amp*1000]),"Diff_messung",np.array([1,2]))
tabelle(np.array([t1_tau*1000,t1_amp*1000]),"T1_messung",np.array([1,2]))
tabelle(np.array([time_a2_cut[extrema_a2]*1000, sig_a2_cut[extrema_a2]*1000]),"T2_messung",np.array([1,2]))


#radius über Molekuelargewicht und dichte
rho = 997
M = 18.01528/1000 / con.N_A

print("Radius über Molekuelargewicht=" ,(M/(rho*4 * np.sqrt(2)))**(1/3) )

b = 31e-6

print("Radius über Van der Waal kovolumen=" ,(3*b/(16*np.pi*con.N_A))**(1/3) )



print("\n\nt2 rel abweichung: ", abs(t2_a - t2_lit) / t2_lit)

print("\n\nt1 rel abweichung: ", abs(-1.0 / t1uparams[0] - t1_lit) / t1_lit)

print("\n\nViskosität rel abweichung: ", abs(viskositaet - visk_lit) / visk_lit)

print("\n\nDiffkonstante rel abweichung: ", abs(duparams[1] - diff_lit) / diff_lit)
