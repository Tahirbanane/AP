import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import sympy
import statistics
from scipy.optimize import curve_fit
from uncertainties import ufloat
from scipy.signal import find_peaks
import pandas as pd

print (f'-----------------------------------Aufgabe a-----------------------------------' )

U, t = np.genfromtxt('a.txt', unpack = True)

U += 4.6
t += 1650
t *= 10**(-6)

def UC(t, U0, RC):
    return U0 * np.exp(-1 * t / RC)

p0=[9.2, 0.00075]
params1, covariance_matrix = curve_fit(UC, t[1:-1], U[1:-1], p0 =p0)
fehler = np.sqrt(np.diag(covariance_matrix))



plt.figure()
plt.plot(t, U,'+', label = 'Spannung U am Kondensator')
plt.plot(t[1:-1], UC(t,params1[0],params1[1])[1:-1])
plt.xlabel('Zeit in s')
plt.ylabel('U in V')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('a.pdf')

print(params1,fehler)



print (f'\n -----------------------------------Aufgabe b-----------------------------------')

f, UC = np.genfromtxt('b.txt', unpack = True)
#f in Hz und Volt

UC /= 2
U0 = 6.88 #Volt. Einmalige Messung

def AC(fr,RC):
    return 1/ (np.sqrt(1+((2*np.pi*f)**2)*(RC**2)))

params, covariance_matrix = curve_fit(AC, f, UC/U0, p0 = [0.0001])
fehler = np.sqrt(np.diag(covariance_matrix))

plt.figure()
plt.plot(f, UC/U0,'+', label = 'Spannung am Kondensator')
plt.plot(f, AC(f, params[0]) ,label = r'$Fit$')
plt.xlabel('Frequenz in s')
plt.ylabel(r'$U_C/U_0$')
plt.xscale('log')
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('b.pdf')

print(params, fehler)

print (f'\n -----------------------------------Aufgabe c-----------------------------------')

f, a, b = np.genfromtxt('c.txt', unpack = True)

phi = a/b * 2 * np.pi

def ph(fr,RC):
    return np.arctan(-2 * np.pi * fr * RC)

params, covariance_matrix = curve_fit(ph, f, phi)
fehler = np.sqrt(np.diag(covariance_matrix))

plt.figure()
plt.plot(f, phi,'+', label = 'Spannung am Kondensator')
plt.plot(f, ph(f, params[0]) ,label = 'Fit')
plt.xlabel('Frequenz in s')
plt.ylabel('Phasenverschiebung')
plt.xscale('log')
plt.legend()
plt.tight_layout()

plt.savefig('c.pdf')

print(params, fehler)

print (f'\n -----------------------------------Aufgabe d-----------------------------------')

U0 = 6.88 #Volt. Einmalige Messung

def AU(fr, ph, RC):
    -np.sin(ph)/(2*np.pi*fr*RC)

RC = -params[0]

AK = 1/ (np.sqrt(1+((2*np.pi*f)**2)*(RC**2)))

plt.figure()
plt.polar(phi, AK, '.', label = 'Messdaten')

x = np.linspace(0, 100000, 10000)
phi = np.arcsin(((x*-params[0])/(np.sqrt(1+x**2*(-params[0])**2)))) #-params[0] weil wert negativ

A = 1/(np.sqrt(1+x**2*(-params[0])**2))
plt.polar(phi, A, label = 'Berechnete Amplitude')
plt.legend()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('d.pdf')