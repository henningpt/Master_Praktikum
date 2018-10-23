import matplotlib.pyplot as plt
import numpy as np


# R_mantel , R_probe = np.genfromtxt()


# t_1= np.linspace(0,x_1,2.5*60)
# t_1= np.linspace(x_1,x_2,5*60)
weiter = 1
fobj = open("messwerte.txt","a")
while(int(weiter)==1):
    R_probe = input("R_probe ? ")
    R_mantel = input("R_mantel ? ")
    fobj.write(str(R_mantel) +"  " + str(R_probe) +"\n" )
    print("\n \n")
    # weiter = input("Weiter 1 Abbrechen 0 ?  ")
fobj.close()
# plt.subplot(1, 2, 1)
# plt.plot(x, y, label='Kurve')
# plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
# plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
# plt.legend(loc='best')
#
# plt.subplot(1, 2, 2)
# plt.plot(x, y, label='Kurve')
# plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
# plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
# plt.legend(loc='best')
#
# # in matplotlibrc leider (noch) nicht möglich
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('build/plot.pdf')
