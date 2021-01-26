import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import sympy
import statistics
from scipy.optimize import curve_fit
from uncertainties import ufloat
from scipy.signal import find_peaks
import scipy.constants as const
import pandas as pd
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)

print (f' -----------------------------------Aufgabe a-----------------------------------')
U, N = np.genfromtxt('Kennlinie.dat', unpack = True)

dN = np.sqrt(N) #Weil Poissonveteilt

Up = U[5:-7]
Np = N[5:-7]

params, covariance_matrix = np.polyfit(Up, Np, deg =1 , cov = True)
errors = np.sqrt(np.diag(covariance_matrix))
x = np.linspace(Up[0], Up[-1])

plt.figure()
plt.errorbar(U,N, xerr = 0, yerr = dN, fmt='--g', label = 'Kennlinie')
plt.plot(x, params[0] * x + params[1], label = 'Plateu-Ausgleichsgerade')
plt.xlabel(r'$U [V]$')
plt.ylabel(r'$N [ Imp / 60 s]$')
plt.legend()
plt.tight_layout()
plt.savefig('a.pdf')

print(f'a = {params[0]} +- {errors[0]}')
print(f'b = {params[1]} +- {errors[1]}')
print(f'Steigung in Prozent pro 100V: {((params[0] * 500 + params[1]) - (params[0] * 400 + params[1])) / 100}')


print (f'\n -----------------------------------Aufgabe b-----------------------------------')

print('ist die Erholungszeit')

print (f'\n -----------------------------------Aufgabe c-----------------------------------')

N1 = 96041.0 / 120.0
N2 = 76518.0 / 120.0
N21 = 158479.0 / 120.0

n1 =  ufloat(N1, np.sqrt(N1))
n2 =  ufloat(N2, np.sqrt(N2))
n12 = ufloat(N21, np.sqrt(N21))

T = (n1 + n2 - n12) / (2 * n1 * n2)

print(f'Totzeit T = {T}')

print (f'\n -----------------------------------Aufgabe d-----------------------------------')

U, N, I0 = np.genfromtxt('Zaehlrohrstrom.dat', unpack = True)
I = unp.uarray(I0, 0.05)
Z = I / (const.epsilon_0 * N)

Zn = unp.nominal_values(Z)
Zs  = unp.std_devs(Z)
plt.figure()
plt.errorbar(I0, Zn, yerr = Zs, fmt = '.')
plt.xlabel('I[μA]') #LaTeX funktioniert nicht
plt.ylabel('Z')
plt.tight_layout()
plt.savefig('d.pdf')