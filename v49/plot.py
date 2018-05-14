import numpy as np
import matplotlib.pyplot as plt

# lade daten
t1_visk = 12.0 * 60.0 + 3.0 + 0.5
t2_visk = 12.0 * 60.0 + 0.3  # zeiten viskosimeter

t1_tau, t1_amp = np.genfromtxt("t1messung.txt", unpack=True)
d_tau, d_amp = np.genfromtxt("diffmessung.txt", unpack=True)
d_amp *= -1.0

print("mittelwert viskosimeter: ", 0.5 * (t1_visk + t2_visk))

# t1 plotten
plt.figure(1)
plt.plot(t1_tau, t1_amp, 'x', label='Messwerte')
plt.xlabel(r'$  \tau \ / \ \mathrm{ms}$')
plt.ylabel(r'$U \ / \ \mathrm{mV}$')
# plt.xscale("log")
plt.legend(loc='best')
plt.show()

# plotten diffusion
plt.figure(2)
plt.plot(d_tau, d_amp, 'x', label='Messwerte')
plt.xlabel(r'$\tau \ / \ \mathrm{ms}$')
plt.ylabel(r'$U \ / \ \mathrm{mV}$')
plt.legend(loc='best')
plt.show()
