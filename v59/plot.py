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

P_mitte = -9.528#in dBm
P_links  = -30.38
P_rechts = -30.37

P_links  =dBm_in_Watt(P_links) #in dBm
P_rechts = dBm_in_Watt(P_rechts)
P_mitte = dBm_in_Watt(P_mitte)

print("Spannung in sqrt(R)", np.sqrt(P_links),np.sqrt(P_mitte),np.sqrt(P_rechts) )

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
