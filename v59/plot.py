import matplotlib.pyplot as plt
import numpy as np
from tabelle import tabelle
import uncertainties.unumpy as unp

#a)


# (b)


# (c)
#modulationsgerad Berechung
print("Aufgabe c)\n\n")
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
# def momentan_f(t,f_t,f_m):
#     return f_t (1 - )
f_T=1e6
f_M=1e5
t_mess=250e-9

# (e)



# (f)


# (g)


# (h)

T=250e-9
x = np.linspace(1e6,5e6,21)
y =np.genfromtxt("messwerte_e.txt",unpack=True)
# x = np.linspace(0, 10, 1000)
# y = x ** np.sin(x)
phase = (x*T*360)%360

tabelle(np.array([x*1e-6,phase,y]),"phasenmessung",np.array([1,1,1]))

plt.plot(phase,y,'x' , label='Messwerte')
plt.legend(loc='best')
# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.tight_layout()
plt.savefig('build/plot.pdf')
