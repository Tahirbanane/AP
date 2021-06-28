import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import scipy.constants as const
from uncertainties import ufloat
from scipy import optimize
from uncertainties.umath import *
from scipy.optimize import fsolve


print (f'\n -----------------------------------Aufgabe a, b-----------------------------------')

p, Tl, Tw = np.genfromtxt('Daten/tiefdruck.txt', unpack = True)

p *= 1e2 # mbar -> Pascal
p0 = 1e5 #atomespheric pressure
Tl += 273.15 # C -> K
Tw += 273.15
R = 8.314462618 #general gas constant

params, covariance_matrix = np.polyfit(1/Tl, np.log(p/p0), deg=1, cov=True)
errors = np.sqrt(np.diag(covariance_matrix))
 
m = ufloat(params[0], errors[0])
b = ufloat(params[1], errors[1])

x_plot = np.linspace(1/Tl[0], 1/Tl[-1], 1000)

plt.plot(1/Tl, np.log(p/p0), 'x', label ='Messwerte')
plt.plot(x_plot, params[0] * x_plot + params[1], label ='Regressionsgerade')

plt.xlabel(r'$\sfrac{1}{T_g} \mathbin{/} \si{\kelvin\tothe{-1}}$')
plt.ylabel(r'$\ln \left ( \sfrac{p}{p_0} \right )$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Daten/tiefdruck.pdf')

L = - m * R #heat of evaporation 

L_i = L - R * 373

L_iMol = L_i / 6.02214076e23 / 1.602e-19

print(f'Die Steigung der Ausgleichsgeraden Tiefdruck ist {m:.2f}')
print(f'Der Achsenabschnitt der Ausgleichsgeraden Tiefdruck ist {b:.2f}')
print(f'Die Verdampfungswärme beträgt {L}')
print(f'L_i beträgt {L_i}')
print(f'L_i pro Molekül beträgt in eV {L_iMol}')


print (f'\n -----------------------------------Aufgabe c, d-----------------------------------')

p, T= np.genfromtxt('Daten/hochdruck.txt', unpack = True)

p *= 1e5 #p in SI
T += 273.15
R = 8.31446261815324 #general gas constant

params, covariance_matrix = np.polyfit(T, p, deg=3, cov=True)
errors = np.sqrt(np.diag(covariance_matrix))
 
a = ufloat(params[0], errors[0])
b = ufloat(params[1], errors[1])
c = ufloat(params[2], errors[2])
d = ufloat(params[3], errors[3])

x_plot = np.linspace(T[0], T[-1], 1000)

plt.plot(T, p, '.', label ='Messwerte')
plt.plot(x_plot, params[0] * x_plot**3 + params[1] * x_plot**2 + params[2] * x_plot + params[3], label ='Regressionsgerade')

plt.xlabel(r'$T \mathbin{/} \si{\kelvin}$')
plt.ylabel(r'$p \mathbin{/} \si{\pascal}$')

plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('Daten/hochdruck.pdf')


print(f'Der Parameter a beim Hochdruck ist {a:.2f}')
print(f'Der Parameter b beim Hochdruck ist {b:.2f}')
print(f'Der Parameter c beim Hochdruck ist {c:.2f}')
print(f'Der Parameter d beim Hochdruck ist {d:.2f}')




