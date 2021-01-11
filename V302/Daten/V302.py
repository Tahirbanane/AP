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
print (f'\n -----------------------------------Aufgabe b-----------------------------------')

C2, R2, R3, R4 = np.genfromtxt('Daten/b.txt', unpack = True)

C2 *= 10**(-9)

def fR(R2,R3,R4):
    return np.sqrt( (R3/R4 * (0.03*R2))**2 + (R2*0.005*R3/R4)**2)

def fC(C2,R3,R4):
    return C2 * 0.005 * R4/R3

print(f'Rx = {R2*R3/R4} +- {fR(R2,R3,R4)} ') #Fehler wurde korrigiert R4/R3 statt R3/R4
print(f'Cx = {C2*R4/R3} +- {fC(C2,R3,R4)} ')

print(f'\n dRx = {ds((R2*R3/R4)[0:3])} +- {ds(fR(R2,R3,R4)[0:3])}') #Fehler wurde korrigiert R4/R3 statt R3/R4
print(f'dCx = {ds((C2*R4/R3)[0:3])} +- {ds(fC(C2,R3,R4)[0:3])}')

print(f'\n dRx = {ds((R2*R3/R4)[3:])} +- {ds(fR(R2,R3,R4)[3:])}') #Fehler wurde korrigiert R4/R3 statt R3/R4
print(f'dCx = {ds((C2*R4/R3)[3:])} +- {ds(fC(C2,R3,R4)[3:])}')
#Aufgbabe c
print (f'\n -----------------------------------Aufgabe c-----------------------------------')

L2, R2, R3, R4 = np.genfromtxt('Daten/c.txt', unpack = True)
L2 *= 10**(-3)

def fL(L2,R3,R4):
    return L2 * 0.005 * R3/R4

print(f'Rx = {R2*R3/R4} +- {fR(R2,R3,R4)}')
print(f'Lx = {L2*R3/R4} +- {fL(L2,R3,R4)}')


print(f'dRx = {ds((R2*R3/R4)[0:3])} +- {ds(fR(R2,R3,R4)[0:3])}')
print(f'dLx = {ds((L2*R3/R4)[0:3])} +- {ds(fL(L2,R3,R4)[0:3])}Wert 10')

print(f'\n dRx = {ds((R2*R3/R4)[3:])} +- {ds(fR(R2,R3,R4)[3:])}')
print(f'dLx = {ds((L2*R3/R4)[3:])} +- {ds(fL(L2,R3,R4)[3:])}Wert 18')

#Aufgabe d
print (f'\n -----------------------------------Aufgabe d-----------------------------------')

R2, R3, R4 = np.genfromtxt('Daten/d.txt', unpack = True)

C4 = 399 * 10**(-9)

def fR(R2,R3,R4):
    return np.sqrt((0.03 * R3/R4)**2 + (R3/((0.003*R4)**2))**2)*R2

print(f'Rx = {R2*R3/R4} +- {fR(R2,R3,R4)}')
print(f'Rx Fehler = {R2*(0.03*R3)/R4 + R2*R3*(0.03*R4)/(R4**2)}')
print(f'Lx = {R2*R3*C4} +- {R2*R3*C4*0.03}')

#Aufgabe e
print (f'\n -----------------------------------Aufgabe e-----------------------------------')

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



plt.figure()
plt.ylabel(r'$U_\text{Br}/U_s$')
plt.xlabel(r'$\omega/\omega_0$')
plt.xscale('log')
plt.plot(w/w0, UBr/Us,'b.' ,label = 'gemessen')
x = np.linspace(0.03,100,1000)
Q = np.sqrt(1/9*((x)**2-1)**2/((1-(x)**2)**2+9*(x)**2))
plt.plot(x, Q,'g--' ,label = 'errechnet')
plt.legend()
plt.tight_layout()
plt.savefig('Daten/grafic.pdf')

x2, _ = find_peaks(-UBr/Us, distance = 15)

print(f'UBr/Us minima: {(w[x2]),w0}')

U2 = UBr/(np.sqrt(1/9*((2)**2-1)**2/((1-(2)**2)**2+9*(2)**2)))

print(f'U2: {ds(U2)}')

k = U2/Us

print(f'k: {k,ds(k)}')

ds(k)