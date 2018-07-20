import matplotlib.pyplot as plt
import numpy as np
from tabelle import tabelle
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
#a)





# (b)

# (c)
print("Aufgabe b)\n\n")
f_M=unp.uarray(100,5)
f_T=unp.uarray(1000,50)
print("theorie Frequenzbander =", f_M+f_T, f_T-f_M )

#modulationsgerad Berechung
print("Aufgabe c)\n\n")

f_M=unp.uarray(100,10)
f_T=unp.uarray(2500,50)
print("theorie Frequenzbander =", f_M+f_M+f_T, f_T-f_M-f_M )




U_delta = unp.uarray(0.27,0.02)
U_T = unp.uarray(2.6,0.1)/2
U_M = unp.uarray(0.5,0.1)/2
print("U_t=" , U_T)
print("U_M=", U_M)
m_c = U_delta / ( U_T * 2 )

print("modulationsgerad bei aufgabenteil aus oszi c)", m_c)
def dBm_in_Watt(x):
    return 10**(x/10)*1e-3

P_mitte = unp.uarray(-9.53,0.05)#in dBm
P_links  = unp.uarray(-30.38,0.05)
P_rechts = unp.uarray(-30.37,0.05)

P_links  =dBm_in_Watt(P_links) #in dBm
P_rechts = dBm_in_Watt(P_rechts)
P_mitte = dBm_in_Watt(P_mitte)


U_mittel_RL= unp.uarray(np.mean([unp.nominal_values((P_rechts)**(1/2)),unp.nominal_values((P_links)**(1/2))]),1/np.sqrt(2)*np.std([unp.nominal_values((P_rechts**(1/2))),unp.nominal_values((P_links)**(1/2))]))
U_mitte = (P_mitte)**(1/2)

print("Spannung in sqrt(R)", (P_links)**(1/2),(P_mitte)**(1/2),(P_rechts)**(1/2))
print("\nMittel_wert von links und Rechts", U_mittel_RL )
print("\nU_mitte=", U_mitte)
m_c_leistung = 2* U_mittel_RL /U_mitte
print("\nModulationsgrad m aus Leistung", m_c_leistung)


modulationsgerad_mittel =   unp.uarray(np.mean([unp.nominal_values(m_c_leistung),unp.nominal_values(m_c)]),1/np.sqrt(2)*np.std([unp.nominal_values(m_c_leistung),unp.nominal_values(m_c)]))
print("\n test::::: mittelwert von beiden modulationsgeraden" ,(m_c_leistung+m_c)/2 )
print("\n mittelwert von beiden modulationsgeraden" ,modulationsgerad_mittel)


# (d)
#frequenzmodulierte schwingung
delta_t= unp.uarray(250e-9,5e-9)

f_T=unp.uarray(1e6,5e4)
f_M=unp.uarray(1e5,1e3)

def modulationsgrad_f(delta_t,f):
    return 2*np.pi*f*delta_t/(3+unp.cos(2*np.pi*f*delta_t ) )

m_d = modulationsgrad_f(delta_t,f_M)
print("\n\n\n d) Modulationsgrad frequenz_Zeit", m_d)



P_mitte_f = unp.uarray( -1.10, 0.05)
P_links_f  = unp.uarray(-14.00,0.05)
P_rechts_f = unp.uarray(-14.04,0.05)

P_links_f  =dBm_in_Watt(P_links) #in dBm
P_rechts_f = dBm_in_Watt(P_rechts)
P_mitte_f = dBm_in_Watt(P_mitte)


U_mittel_RL_f= unp.uarray(np.mean([unp.nominal_values((P_rechts_f)**(1/2)),unp.nominal_values((P_links_f)**(1/2))]),1/np.sqrt(2)*np.std([unp.nominal_values((P_rechts_f**(1/2))),unp.nominal_values((P_links_f)**(1/2))]))
U_mitte_f = (P_mitte_f)**(1/2)

print("Spannung in sqrt(R)", (P_links_f)**(1/2),(P_mitte_f)**(1/2),(P_rechts_f)**(1/2))
print("\nMittel_wert von links und Rechts", U_mittel_RL_f )
print("\nU_mitte=", U_mitte_f)
m_d_leistung_f = 2* U_mittel_RL_f /U_mitte_f *(f_M/f_T)
print("\nModulationsgrad m aus Leistung", m_d_leistung_f)

modulationsgerad_mittel =   unp.uarray(np.mean([unp.nominal_values(m_d_leistung_f),unp.nominal_values(m_d)]),1/np.sqrt(2)*np.std([unp.nominal_values(m_d_leistung_f),unp.nominal_values(m_d)]))
print("\n test::::: mittelwert von beiden modulationsgeraden" ,(m_d_leistung_f+m_d)/2 )
print("\n mittelwert von beiden modulationsgeraden" ,modulationsgerad_mittel)




# (e)



# (f)

# (g)


# (h)

def cos_fit(x,a,phi):
    return a*np.cos(np.pi * x/180+phi)

T=250e-9
x = np.linspace(1e6,5e6,21)
x_lim = np.linspace(0,360,10000)
y =np.genfromtxt("messwerte_e.txt",unpack=True)
# x = np.linspace(0, 10, 1000)
# y = x ** np.sin(x)
phase = (x*T*360)%360
params, cov =curve_fit(cos_fit,phase,y,p0=[67,np.pi])
uparams = unp.uarray(params, np.sqrt(np.diag(cov)))
print("\n\n fitparameter", uparams)
tabelle(np.array([x*1e-6,phase,y]),"phasenmessung",np.array([1,1,1]))
plt.plot(x_lim, cos_fit(x_lim,*params),label=r"Fit")
plt.plot(phase, y, 'x' , label=r'Messwerte')
plt.legend(loc='best')
plt.xlabel(r"$\phi / \si{\degree}$")
plt.ylabel(r"$U_G(\phi) / \si{\milli\volt}$")
# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.tight_layout()
plt.savefig('build/plot.pdf')
