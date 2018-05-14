import numpy as np
import matplotlib.pyplot as plt

# lade daten
t1_tau, t1_amp = np.genfromtxt("t1messung.txt", unpack=True)

print("test", t1_amp)
# plotten
plt.figure(1)
plt.plot(t1_tau, t1_amp, 'x', label='Messwerte')
plt.xlabel(r'$\ln\tau \ / \ \mathrm{ms}$')
plt.ylabel(r'$U \ / \ \mathrm{mV}$')
plt.xscale("log")
plt.legend(loc='best')
plt.show()
