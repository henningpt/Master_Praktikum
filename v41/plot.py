import matplotlib.pyplot as plt
import numpy as np


# lade daten
r_meta = np.genfromtxt("metall.txt", unpack=True)
r_salz = np.genfromtxt("salz.txt", unpack=True)


# groessen definieren
wavelen = 1.5417e-10
camera_rad = 57.3 * 10**(-3)
d_probe = 130 * 10**(-3)


# functions
def winkel(d, R):
    return (R / (2 * d))


def test(m_sum, lamb, a):
    m_sum = np.sqrt(m_sum)
    return(180 * np.arcsin(m_sum * lamb/(2 * a)) / np.pi)


print("Dies ist ein Test: ", test(18, wavelen, 3e-10))


# rechnen
# richtige radien
r_meta = r_meta + 0.9
r_salz = r_salz + 0.9

# berechne zugehoerige winkel theta
salz_winkel = winkel(d_probe, r_salz)
meta_winkel = winkel(d_probe, r_meta)


# auf konsole ausgeben
print("\n\n\nwinkel salz:\n")
for element in salz_winkel:
    print(element)

print("\n\n\nwinkel metall:\n")
for element in meta_winkel:
    print(element)

# x = np.linspace(0, 10, 1000)
# y = x ** np.sin(x)
#
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
