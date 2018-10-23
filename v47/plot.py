import numpy as np
import matplotlib.pyplot as plt

def Temperatur(R):
    return(0.00133 * R**2 + 2.296 * R  - 243.02)


def Molwaerme_druck(delta_Q,masse,Delta_T):
        return delta_Q/(masse*Delta_T)

R_mantel , R_probe = np.genfromtxt("messwerte.txt", unpack=True)

masse_Cu = 0.342



t_1 = np.linspace(0,22.5,10)
t_2 = np.linspace(25, 32.5,len(R_probe)-10)



t_ges = np.append(t_1,t_2)
print(t_ges)
plt.figure(1)
plt.plot(t_ges, Temperatur(R_probe),'x' ,label='Probe')
plt.plot(t_ges, Temperatur(R_mantel),'x' ,label='Mantel')
plt.legend(loc='best')
plt.savefig('build/temperatur_verlauf.pdf')

plt.figure(2)
plt.plot(t_ges, Temperatur(R_mantel)-Temperatur(R_probe),'x' ,label='Mantel')
plt.savefig('build/temperatur_diff.pdf')
plt.close

plt.figure(3)
plt.plot(Temperatur(R_probe[1:]),np.append(Molwaerme_druck(1,masse_Cu,Temperatur(R_probe[1:10])-Temperatur(R_probe[:9])),Molwaerme_druck(2,masse_Cu,Temperatur(R_probe[10:])-Temperatur(R_probe[9:-1]))),'x' ,label='C_p')
plt.savefig('build/molwaerme_druck_const.pdf')
plt.close

index = np.array([0,2,4,6,8,10])

index1 = np.array([2,4,6,8,10])
index2 = np.array([0,2,4,6,8])
print(np.append(Temperatur(R_probe[index]),Temperatur(R_probe[9:])))


plt.figure(4)
plt.plot(np.append(Temperatur(R_probe[index]),Temperatur(R_probe[11:])) ,np.append(Molwaerme_druck(1,masse_Cu,Temperatur(R_probe[index1])-Temperatur(R_probe[index2])),Molwaerme_druck(2,masse_Cu,Temperatur(R_probe[10:])-Temperatur(R_probe[9:-1]))),'x' ,label='C_p')
plt.savefig('build/molwaerme_druck_const(2).pdf')
plt.close



# plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
# plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
#
# # in matplotlibrc leider (noch) nicht möglich
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
