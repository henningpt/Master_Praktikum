import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1e6,5e6,21)
y =np.genfromtxt("messwerte_e.txt",unpack=True)
# x = np.linspace(0, 10, 1000)
# y = x ** np.sin(x)



plt.plot(x,y,'x' , label='Kurve')
plt.legend(loc='best')


# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')
