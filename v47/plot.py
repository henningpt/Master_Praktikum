import numpy as np
import matplotlib.pyplot as plt
from tabelle import tabelle
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import scipy.constants as const
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)

def omega_D(m, rho, molare_masse):
    v_l  = 4.7e3   # in Meter/sekunde
    v_tr = 2.26e3   # in Meter/sekunde
    # N_L  = m / (Atommasse * const.physical_constants['atomic mass constant'][0])
    N_L = (m / molare_masse) * const.N_A
    Volumen = m / rho
    omega = (18 * np.pi**2 * N_L / (Volumen * ( 1/v_l**3 + 2/(v_tr**3) )) )**(1/3)
            # ((18*np.pi**2*teilchenzahl)/volumen*1/((1/vlong**3)+(2/vtrans**3)))**(1/3)
    print("Omega_D = ", omega )
    print("Temp_D = ", const.hbar * omega / const.k )
    return const.hbar * omega / const.k

def Temperatur(R):
    return(0.00133 * R**2 + 2.296 * R  - 243.02)

def fit_polynom_grad4(x,a,b,c,d,e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def fit_polynom_grad7(x,a,b,c,d,e,f,g):
    # return a * x **4 + b * x **3 + c * x**2 + d * x + e + f/x
    return a * x **3 + b * x **2 + c * x   + d + e/x  + f * x**(-2) + g * x**(-3)

def Molwaerme_druck(U,I,masse,Delta_T,delta_t):
        print("Delta_T=", Delta_T)
        print("delta_t=", delta_t)
        CU_molvol = 7.11e-6  #http://www.chemie.de/lexikon/Kupfer.html
        CU_dichte = 8920 # http://www.chemie.de/lexikon/Kupfer.html
        masse_mol = masse/(CU_dichte * CU_molvol)
        return U*I*delta_t/(masse_mol*Delta_T)

def Molwaerme_volumen(C_P, alpha, kompress, molvol, T):
            C_V = (C_P - 9*alpha**2 * kompress * molvol * T)
            print("Cv-Cp",C_V-C_P)
            return (C_V)


alpha_CU = np.array([7.00, 8.50, 9.75, 10.70, 11.50, 12.10, 12.65,
13.15, 13.60, 13.90, 14.25, 14.50, 14.75, 14.95, 15.20, 15.40,
15.60, 15.75, 15.90, 16.10, 16.25, 16.35, 16.50, 16.65]) *1e-6
T_alpha = np.linspace(70,300,24)

params_alpha , cov_alpha = curve_fit(fit_polynom_grad4, T_alpha , alpha_CU)
uparams_alpha = unp.uarray(params_alpha, np.sqrt(np.diag(cov_alpha)))

print("fitparams alpha", uparams_alpha)
T_alpha_fit = np.linspace(70,300,50)

R_mantel , R_probe = np.genfromtxt("messwerte.txt", unpack=True)


masse_Cu = 0.342



t_1 = np.linspace(0,22.5,10)
t_2 = np.linspace(25, 25+5*float(len(R_probe)-11),len(R_probe)-10)
print(len(R_probe)-10)
I=unp.uarray(152,2)*1e-3
U=unp.uarray(16.0,0.5)
CU_molvol = 7.11e-6  #http://www.chemie.de/lexikon/Kupfer.html
CU_kompress = 140e9 #Pa  http://www.periodensystem-online.de/index.php?show=list&id=modify&prop=Kompressionsmodul&sel=oz&el=68




t_ges = np.append(t_1,t_2)*60
print(t_ges)

T_probe = Temperatur(R_probe) + const.zero_Celsius
T_mantel = Temperatur(R_mantel)  + const.zero_Celsius
print("T_probe=", T_probe)
T_diff = T_probe[1:]-T_probe[:-1]
t_diff = t_ges[1:]-t_ges[:-1]




C_P = Molwaerme_druck(U,I,masse_Cu,T_probe[1:]-T_probe[:-1], t_ges[1:]-t_ges[:-1])


plt.figure(1)
plt.plot(t_ges, T_probe,'x' ,label='Probe')
plt.plot(t_ges, T_mantel,'x' ,label='Mantel')
plt.xlabel("Zeit t/s")
plt.ylabel("Temperatur T/K")
plt.legend(loc='best')
plt.savefig('build/temperatur_verlauf.pdf')

plt.figure(2)
plt.plot(t_ges, Temperatur(R_mantel)-Temperatur(R_probe),'x' ,label='Mantel')
plt.savefig('build/temperatur_diff.pdf')
plt.close


T_mittel = unp.uarray(T_probe[:-1] + T_diff/2, T_diff/2)
plt.figure(3)
plt.errorbar(T_probe[:-1] + T_diff/2 ,noms(C_P),xerr = T_diff/2 , yerr = stds(C_P), fmt='x' )
plt.savefig('build/molwaerme_druck_const.pdf')
plt.close

plt.figure(4)
plt.plot(T_alpha, alpha_CU ,'x' ,label='Alpha')
plt.plot(T_alpha_fit, fit_polynom_grad4(T_alpha_fit,*params_alpha))
plt.savefig('build/alpha_CU.pdf')
plt.close

C_V = Molwaerme_volumen(C_P, fit_polynom_grad4(T_mittel,*params_alpha), CU_kompress, CU_molvol, T_mittel)



tabelle(np.array([t_ges ,R_probe ,T_probe , R_mantel, T_mantel,T_mantel-T_probe]),
    "messwerte_tabelle",np.array([0,1,1,1,1,1]))

print("T_mantel-T_probe", np.max(abs(T_mantel-T_probe)))

alpha_Messung = fit_polynom_grad4(T_mittel,*params_alpha)
print('test',np.array([T_mittel, alpha_Messung]))


tabelle(np.array([t_diff , T_diff, noms(C_P), stds(C_P), noms(T_mittel), stds(T_mittel), noms(alpha_Messung)*1e6, stds(alpha_Messung)*1e6, noms(C_V), stds(C_V) ]),
    "Messwerte_C_P_C_V", np.array([0,1,2,2,1,1,2,2,2,2]))



plt.figure(5)
plt.plot(noms(T_mittel),noms(T_mittel)*0+3*const.R,'--b')
plt.errorbar(T_probe[:-1] + T_diff/2 ,noms(C_P),xerr = T_diff/2 , yerr = stds(C_P), fmt='x', label="C_P" )
plt.errorbar(noms(T_mittel),noms(C_V), xerr = stds(T_mittel), yerr = stds(C_V), fmt='x', label="C_V")
plt.legend(loc='best')
plt.savefig('build/molwaerme_volumen_const.pdf')
plt.close

# Berechnung von Debye Temperatur

print("test", T_mittel[T_mittel<170])
print(C_V[T_mittel<170])

T_170 = T_mittel[T_mittel<170]
C_V_170 = C_V[T_mittel<170]

debyefunktion_C_V =  np.reshape(np.genfromtxt("debyefunktion.txt"),160)
debyefunktion_O_T = np.linspace(0,15.9,160)

print(debyefunktion_C_V)
print(debyefunktion_O_T)
debye_temp_lit = 345

C_V_170 = np.delete(C_V_170,[2,9],None)
T_170 = np.delete(T_170,[2,9],None)

plt.figure(10)
plt.plot(debye_temp_lit/debyefunktion_O_T[10:],3*const.R+debye_temp_lit/debyefunktion_O_T[10:]*0,'-',label=r'Klassisch')
plt.errorbar(T_probe[:-1] + T_diff/2 ,noms(C_V),xerr = T_diff/2 , yerr = stds(C_V), fmt='x',label=r'Messwerte', zorder=-1 )
plt.plot(debye_temp_lit/debyefunktion_O_T[10:], debyefunktion_C_V[10:], "-", label=r"Debyemodell")
plt.plot(noms(T_170), noms(C_V_170), '.', label=r"Verwendete Messwerte")
plt.xlabel(r"$T/K$")
plt.ylabel(r"$C_V /  [C_V]$")
plt.legend(loc="best")
plt.savefig('build/molwaerme_V_const_und_theo.pdf')
plt.close




params_debye , cov_debye = curve_fit(fit_polynom_grad4, debyefunktion_C_V[(debyefunktion_C_V<22) & (debyefunktion_C_V>14)] , debyefunktion_O_T[(debyefunktion_C_V<22) & (debyefunktion_C_V>14)] )
uparams_debye = unp.uarray(params_debye, np.sqrt(np.diag(cov_debye)))
print("debyefitparameter", uparams_debye)


plt.figure(6)
plt.plot(debyefunktion_C_V,debyefunktion_O_T ,'x' ,label=r'Debyefunktion')
plt.plot(debyefunktion_C_V, fit_polynom_grad4(debyefunktion_C_V,*params_debye),label = r'Fitfunktion')
plt.xlabel(r'$C_V/ [C_V]}$')
plt.ylabel(r'$\frac{\theta_D}{T}$')
plt.legend(loc='best')
plt.savefig('build/debyefunktion.pdf')
plt.close

Gemessen_O_T = fit_polynom_grad4(C_V_170,*params_debye)
print("Test",Gemessen_O_T)
print("parmas mit fehler", uparams_debye )
debye_temperatur_kupfer = Gemessen_O_T * T_170
print('debye Temperatur', debye_temperatur_kupfer)
print('mittelwert debye Temperatur', np.mean(noms(debye_temperatur_kupfer) ), np.std(noms(debye_temperatur_kupfer) ) )
T_D_mittel = unp.uarray(np.mean(noms(debye_temperatur_kupfer)),np.std(noms(debye_temperatur_kupfer) ) )


tabelle(np.array([noms(C_V_170),stds(C_V_170), noms(Gemessen_O_T), stds(Gemessen_O_T), noms(T_170), stds(T_170), noms(debye_temperatur_kupfer), stds(debye_temperatur_kupfer)]),
    "temperatur_debye",np.array([2,2,2,2,1,1,1,1]))

#theoriewert
CU_rho = 8.96e3    # wolfram
CU_Molmasse = 63.546e-3 #wolfram
T_D = omega_D(masse_Cu , CU_rho , CU_Molmasse)
print("relative Abweichung zum Theoriewert", (T_D-T_D_mittel)/T_D )
T_D_lit= 345
print("relative Abweichung zum Litwert", (T_D_lit-T_D_mittel)/T_D_lit )
print("relative Abweichung zum Litwert des Theoriewertes", (T_D_lit-T_D)/T_D_lit )

# plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
# plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
#
# # in matplotlibrc leider (noch) nicht möglich
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
