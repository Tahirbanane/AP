import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import sympy
import statistics
from scipy.optimize import curve_fit
from uncertainties import ufloat

def ds(A):
    return statistics.mean(A)


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

C4 = 399 * 10**(-9)

print(f'Rx = {R2*R3/R4}')
print(f'Rx Fehler = {R2*(0.03*R3)/R4 + R2*R3/((0.03*R4)**2)}')
print(f'Lx = {R2*R3*C4} +- 3% Fehler')

#Aufgabe e


