import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import uncertainties.unumpy as unp
import sympy
import statistics
from scipy.optimize import curve_fit
from uncertainties import ufloat
from scipy.signal import find_peaks
import scipy.constants as const

e=const.e
h=const.h
c=const.c
d_LiF=201.4*10**(-12) #m
ry=13.6 #eV
alpha=const.value("fine-structure constant")

theta_cu,N_cu = np.genfromtxt('EmissionCu.txt', unpack = True)
N_max = find_peaks(N_cu, height=1000)
N_peak_cu = N_cu[N_max[0]]
theta_peak_cu = theta_cu[N_max[0]]
plt.plot(theta_cu, N_cu,'kx',label = 'Messdaten')
plt.ylabel(r'$N \,/\, \mathrm{\frac{Imp}{s}}$')
plt.plot(theta_cu[0:120],N_cu[0:120], color='orange',ls='--',label = 'Bremsberg')
plt.plot(theta_cu[127:143],N_cu[127:143],color='orange',ls='--')
plt.plot(theta_cu[150:180],N_cu[150:180],color='orange',ls='--')
plt.axvline(theta_peak_cu[0], color='steelblue', ls = '--', label = r'$K_{\mathrm{\beta}}$')
plt.axvline(theta_peak_cu[1], color='g', ls = '--', label = r'$K_{\mathrm{\alpha}}$')
plt.xlabel(r'$\theta \,/\, °$')
plt.legend()
plt.tight_layout()
plt.savefig("emissionsspektrum.pdf")
plt.close()

print(f"k-linien bei {theta_peak_cu}°")

E_K_peaka = (h*c)/(2*d_LiF*np.sin(theta_peak_cu[1]*np.pi/180))
E_K_peaka=E_K_peaka/const.e #eV
E_K_peakb = (h*c)/(2*d_LiF*np.sin(theta_peak_cu[0]*np.pi/180))
E_K_peakb=E_K_peakb/const.e #eV

print(f"Energien: E_a = {E_K_peaka} E_b = {E_K_peakb}")

a_o, N_o_ = np.genfromtxt('ComptonOhne.txt', unpack=True)
a_al, N_al_ = np.genfromtxt('ComptonAl.txt', unpack=True)
Tau = 90*10**(-6)

N_o_err = np.sqrt(N_o_)
N_al_err = np.sqrt(N_al_)

N_o = unp.uarray(N_o_, N_o_err)
N_al = unp.uarray(N_al_, N_al_err)

# Totzeitkorrektur
I_o = N_o / (1 - Tau * N_o)
I_al = N_al / (1 - Tau * N_al)

T = I_al / I_o
lam = 2 * d_LiF * np.sin(a_o * np.pi / 180)

params, covariance_matrix = np.polyfit(lam, unp.nominal_values(T), deg=1, cov=True)

errors = np.sqrt(np.diag(covariance_matrix))
m = ufloat(params[0], errors[0])
b = ufloat(params[1], errors[1])
print(f"m = {m}")
print(f" b = {b}")
x_plot = np.linspace(np.min(lam),np.max(lam),100)
plt.plot(x_plot, params[0] * x_plot + params[1], color='steelblue', ls= '-', label='Lineare Regression', linewidth=2,)
plt.plot(lam, unp.nominal_values(T),'kx',label = 'Messdaten')
plt.errorbar(lam, unp.nominal_values(T), yerr=unp.std_devs(T), fmt='g_', label='Fehler')
plt.xlabel(r'$\lambda \,/\, \mathrm{m}$')
plt.ylabel(r'$T(\lambda)$')
plt.legend()
plt.tight_layout()
plt.savefig("transmission.pdf")
plt.close()

I_0 = 2731
I_1 = 1180
I_2 = 1024

T_1 = I_1/I_0
T_2 = I_2/I_0

lam_1 = (T_1 - b )/m
lam_2 = (T_2 - b )/m
lam_c = lam_2 - lam_1
lam_c_theo = 2.4263*10**(-12)

print(f"T_1 = {T_1} \nT_2 = {T_2 }\nlam_1 = {lam_1} \nlam_2 = {lam_2} \nlam_c = {lam_c}")

rel_lam_c = (lam_c-lam_c_theo)/lam_c_theo
print(f"rel_lam_c = {rel_lam_c}")