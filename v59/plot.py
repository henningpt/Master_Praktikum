import matplotlib.pyplot as plt
import numpy as np
from tabelle import tabelle
# (a)


# (b)


# (c)


# (d)


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
