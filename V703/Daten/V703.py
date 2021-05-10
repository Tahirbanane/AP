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

y=params[0] * 370 + params[1]
z=params[0] * 570 + params[1]
print((0.5*(z-y))/y)


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

U, N, I = np.genfromtxt('Zaehlrohrstrom.dat', unpack = True) # I in µA
I *= 10**(-6) # I in A 
N /= 60

Nerr = unp.uarray(N, np.sqrt(N))

Ierr = unp.uarray(I, 0.05 * 10**(-6))
Z = Ierr / (const.elementary_charge * Nerr)

I *= 10**6 # I in µA

Zoerr = np.array([11420876622.984013, 14987115336.374016, 25540080000.716293, 29513588372.979668, 36772441522.746056, 47482464430.69732, 49965382850.9200,  58377325715.92296])
Zerr  = np.array([1903479437.1640024, 1873389417.046752, 1824291428.6225922, 1844599273.3112292, 1838622076.13730264, 1826248631.9498968, 1784477958.9614303, 1621592380.99786])

print(Z * 10**(-9))

Zoerr *= 10**(-9)
Zerr  *= 10**(-9)
plt.figure()
plt.errorbar(I, Zoerr, yerr = Zerr, fmt = '.')
plt.xlabel('I[μA]') #LaTeX funktioniert nicht
plt.ylabel('Z')
plt.tight_layout()
plt.savefig('d.pdf')
