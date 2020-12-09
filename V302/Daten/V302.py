import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import sympy
import statistics
from scipy.optimize import curve_fit
from uncertainties import ufloat
from scipy.signal import find_peaks

def ds(A):
    return statistics.mean(A)

from functools import reduce

def get_minmax(values):
    return reduce(
        lambda mm, v: (min(v, mm[0]), max(v, mm[1])),
        values,
        (values[0],) * 2
    )

#Aufgabe a bereits händisch berechnet


#Aufgabe b

C2, R2, R3, R4 = np.genfromtxt('Daten/b.txt', unpack = True)

C2 *= 10**(-9)

print(f'Rx = {R2*R3/R4} +- 3% Fehler')
print(f'Cx = {C2*R3/R4}')

#Aufgbabe c

L2, R2, R3, R4 = np.genfromtxt('Daten/c.txt', unpack = True)

L2 *= 10**(-3)

print(f'Rx = {R2*R3/R4}')
print(f'Lx = {L2*R3/R4}')

#Aufgabe d

R2, R3, R4 = np.genfromtxt('Daten/d.txt', unpack = True)

C4 = 399 * 10**(-9)

print(f'Rx = {R2*R3/R4}')
print(f'Rx Fehler = {R2*(0.03*R3)/R4 + R2*R3*(0.03*R4)/(R4**2)}')
print(f'Lx = {R2*R3*C4} +- 3% Fehler')

#Aufgabe e

# R = 1000 Ohm
# w = 2pi f
f, UBr, Us = np.genfromtxt('Daten/e.txt', unpack = True)

UBr /= 2
Us /= 2

w = 2*np.pi*f
R = 1000
C = (399 + 450)/2 #mittelwert der beiden Kondensatoren
C *= 10**(-9)

w0 = 1/(R*C)

Q = np.sqrt(1/9*((w/w0)**2-1)**2/((1-(w/w0)**2)**2+9*(w/w0)**2))

plt.figure()
plt.ylabel(r'$U_\text{Br}/U_s$')
plt.xlabel(r'$\omega/\omega_0$')
plt.xscale('log')
plt.plot(w/w0, UBr/Us,'b.' ,label = r'$gemessen$')
plt.plot(w/w0, Q,'g--' ,label = r'$errechnet$')
plt.legend()
plt.tight_layout()
plt.savefig('Daten/grafic.pdf')

x2, _ = find_peaks(-UBr/Us, distance = 15)

print(f'UBr/Us minima: {get_minmax(UBr/Us[x2])}')

U2 = UBr/(np.sqrt(1/9*((2)**2-1)**2/((1-(2)**2)**2+9*(2)**2)))

print(f'U2: {U2}')

k = U2/Us

print(f'k: {k,ds(k)}')

ds(k)