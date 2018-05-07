import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit as cf
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)
# from tabelle.py import tabelle

# lade daten
r_meta = np.genfromtxt("metall.txt", unpack=True)


bcc= np.array([[0, 1, 1],#2
                [0, 0, 2],#4
                [1, 1, 2],#6
                [0, 2, 2],#8
                [0, 1, 3],#10
                [2, 2, 2],#12
                [1, 2, 3],#14
                [1, 1, 4],#18
                ])

fcc=np.array([[1, 1, 1],#3
              [0, 0, 2],#4
              [0, 2, 2],#8
              [1, 1, 3],#11
              [2, 2, 2],#12
              [0, 0, 4],#16
              [1, 3, 3],#19
              [0, 2, 4],#20
 ])


dia = np.array([[1, 1, 1],#3
              [0, 2, 2],#8
              [1, 1, 3],#11
              [0, 0, 4],#16
              [1, 3, 3],#19
              [2, 2, 4],#24
              [3, 3, 3],#27
              [4, 4, 0],#32
 ])

bcc=bcc.astype(np.float64)
fcc=fcc.astype(np.float64)
dia=dia.astype(np.float64)

# groessen definieren
wavelen = 1.5417e-10
camera_rad = 57.3 * 10**(-3)
d_probe = 130 * 10**(-3)
proben_rad = 0.001  # nicht endgültig
v = 0.002


# functions
def funp(string, arr):
    print("\n\n\n" + string)
    for element in arr:
        print(element)


def winkel(d, R):
    return (R / (2 * d))


def inrad(winkel_deg):
    return(np.pi * winkel_deg / 180)


def test(m_sum, lamb, a):
    m_sum = np.sqrt(m_sum)
    return(180 * np.arcsin(m_sum * lamb/(2 * a)) / np.pi)


# Gitterkonstante für jeden winkel berechnen
def gitter(gitter, lamb, theta):
    m_sum = miller(gitter)

    print("\n\n\nm_sum aus miller aus gitter", m_sum)
    return(m_sum * lamb / (2 * np.sin(inrad(theta))))


# Betragsquadrat der Millerindices
def miller(struktur):
    mill_arr = np.array(struktur[:,1])

    for ind, val in enumerate(struktur[:, 1]):
        mill_arr[ind] = np.linalg.norm(struktur[ind, :])
        print("\nstruktur ", struktur[ind,:])
    return(mill_arr)

def gk_korrektur_a(gk, theta, proben_rad, camera_rad):
    return(gk * proben_rad * (1 - camera_rad) * np.cos(inrad(theta))**2 / (2 * camera_rad * inrad(theta)))


def winkel_korrektur(theta):
    lam = 1.5417*10**(-10)
    lam_1 = 1.54093*10**(-10)
    lam_2 = 1.54478*10**(-10)
    return ((lam_1-lam_2)*np.tan(inrad(theta))/lam)


def gk_korrektur_v(gk, theta, camera_rad, v):
    return(gk * v * np.cos(inrad(theta))**2 / (camera_rad))
# print("Dies ist ein Test: ", test(18, wavelen, 3e-10))


# fitten
def gk_plot(name, winkel, gk, funktion, korr, fitgrenzen):
    params, errors = cf(funktion, np.cos(inrad(winkel[fitgrenzen[0]:fitgrenzen[1]]))**2,
                                  noms(gk[fitgrenzen[0]:fitgrenzen[1]]))
    g_plot = np.linspace(0.1, max(np.cos(inrad(winkel)**2)))
    plt.errorbar(np.cos(inrad(winkel))**2, noms(gk), xerr=korr(winkel),
                 yerr=stds(gk), fmt='x')
    plt.plot(g_plot, funktion(g_plot, *params))
    plt.savefig("build/plot_"+name+".pdf")
    plt.close()
    return(unp.uarray(params[1], np.diag(errors)[1]))



# RECHNEN
# richtige radien
r_meta = r_meta + 0.9/2

# berechne zugehoerige winkel theta
meta_winkel = winkel(d_probe, r_meta)

# gitterkonstanten bcc metall
gk_m_bcc = gitter(bcc, wavelen, meta_winkel)
gk_m_fcc = gitter(fcc, wavelen, meta_winkel)
gk_m_dia = gitter(dia, wavelen, meta_winkel)


a_m_bcc=unp.uarray(gk_m_bcc,gk_korrektur_a(gk_m_bcc, meta_winkel, proben_rad, camera_rad) + gk_korrektur_v(gk_m_bcc, meta_winkel, camera_rad, v))
a_m_fcc=unp.uarray(gk_m_fcc,gk_korrektur_a(gk_m_fcc, meta_winkel, proben_rad, camera_rad) + gk_korrektur_v(gk_m_fcc, meta_winkel, camera_rad, v))
a_m_dia=unp.uarray(gk_m_dia,gk_korrektur_a(gk_m_dia, meta_winkel, proben_rad, camera_rad) + gk_korrektur_v(gk_m_dia, meta_winkel, camera_rad, v))

# auf konsole ausgeben

print("\n\n\nwinkel metall:\n")
for element in meta_winkel:
    print(element)

print("\n\n\nGitterkonstanten bcc metall: ")
for element in gk_m_bcc:
    print(element)


print("\n\n\nGitterkonstanten fcc metall: ")
for element in gk_m_fcc:
    print(element)


print("\n\n\nGitterkonstanten dia metall: ")
for element in gk_m_dia:
    print(element)

funp(" Korrekturen bcc: ", a_m_bcc)


def gerade(x,a,b):
    return (a*x+b)



# fit plto fuer metall bcc
a_bcc_end=gk_plot("bcc", meta_winkel, a_m_bcc, gerade, winkel_korrektur,[0,7])
a_fcc_end=gk_plot("fcc", meta_winkel, a_m_fcc, gerade, winkel_korrektur,[0,7])
a_dia_end=gk_plot("dia", meta_winkel, a_m_dia, gerade, winkel_korrektur,[1,7])

print("\nMetall bcc",a_bcc_end)
print(" \nMetall fcc",a_fcc_end)
print(" \nMetall dia",a_dia_end)



#TESTESTEST
print("\n\n\n\nasldkfjölaksdflj", test(3, wavelen, 2.8665 * 10**(-10)))
##################################################################################
#salz

# berechne zugehoerige winkel theta
r_salz = np.genfromtxt("salz.txt", unpack=True)

# richtige radien
r_salz = r_salz + 0.9/2

salz_winkel = winkel(d_probe, r_salz)

print("\n\n\nwinkel salz:\n")
for element in salz_winkel:
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
