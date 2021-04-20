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


print (f'\n -----------------------------------Aufgabe a-----------------------------------')
#Diode 1
U1, I1 = np.genfromtxt('Daten/2,1A.txt', unpack = True)

plt.figure()
plt.plot(U1, I1, 'r.' ,label = 'Messung bei 2.1A')
k = np.linspace(0.27,0.27,14)
print('2.1A: 0.27 mA')
plt.plot(U1,k, 'r--' , label = r' $I_1$ bei 2.1A')

U2, I2 = np.genfromtxt('Daten/2,2A.txt', unpack = True)
plt.plot(U2, I2,'y.' ,label = 'Messung bei 2.2A')
k = np.linspace(0.6,0.6,14)
print('2.2A: 0.6 mA')
plt.plot(U2,k, 'y--' , label = r' $I_2$ bei 2.2A')

U3, I3 = np.genfromtxt('Daten/2,3A.txt', unpack = True)
plt.plot(U3, I3, 'g.' ,label = 'Messung bei 2.3A')
k = np.linspace(1.2,1.2,14)
print('2.3A: 1.2 mA')
plt.plot(U3,k, 'g--' , label = r' $I_3$ bei 2.3A')

U4, I4 = np.genfromtxt('Daten/2,4A.txt', unpack = True)
plt.plot(U4, I4, 'c.' ,label = 'Messung bei 2.4A')
k = np.linspace(2.3,2.3,14)
print('2.4A: 2.3 mA')
plt.plot(U4,k, 'c--' , label = r' $I_4$ bei 2.4A')

U5, I5 = np.genfromtxt('Daten/2,5A.txt', unpack = True)
plt.plot(U5, I5,'k.' ,label = 'Messung bei 2.5A')
k = np.linspace(3.5,3.5,26)
print('2.5A: 3.5 mA')
plt.plot(U5,k, 'k--' , label = r' $I_5$ bei 2.5A')

plt.xlabel(r'$U \: / \: V$')
plt.ylabel(r'$I \: / \: mA$')
plt.tight_layout()
legend = plt.legend(loc="upper left", edgecolor="grey")
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((1, 1, 1, 1))
plt.grid(True)
plt.savefig('Daten/kennlinie.pdf')

print (f'\n -----------------------------------Aufgabe b-----------------------------------')

U, I = np.genfromtxt('Daten/2,5A.txt', unpack=True)

def f (V, c, d):
    return c* V**d

params, covariance_matrix = curve_fit(f, U[1:-10], I[1:-10])
uncertainties = np.sqrt(np.diag(covariance_matrix))

plt.figure()
plt.plot(U, f(U, *params), label ='errechnet')
plt.plot(U, I, '.', label ='gemessen')

plt.xlabel(r'$U \: / \: V$')
plt.ylabel(r'$I \: / \: mA$')

plt.xlim(np.min(U)-0.5, np.max(U)+0.5)
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.savefig('Daten/exponent.pdf')

print(f' c= {params[0]} pm {uncertainties[0]}')
print(f' d= {params[1]} pm {uncertainties[1]}')

print (f'\n -----------------------------------Aufgabe c-----------------------------------')

U, I = np.genfromtxt('Daten/gegenfeld.txt', unpack=True)

#Anpassung des Stroms wegen dem 1MOhm Widerstand
Uk = U + I / 10**(3) # U = RI und I ist in nano Ampere

#def g(V, x, T):
#   return x * np.exp(-1 * V * const.epsilon_0/(T * const.k))
#
#print(const.epsilon_0 + 1)
#
#params, covariance_matrix = curve_fit(g, Uk, I)
#uncertainties = np.sqrt(np.diag(covariance_matrix))

params, covariance_matrix = np.polyfit(Uk, np.log(I), deg=1, cov=True)
errors = np.sqrt(np.diag(covariance_matrix))

plt.figure()
plt.plot(Uk, params[0]* Uk + params[1], label ='errechnet')
plt.plot(Uk, np.log(I), '--', label ='gemessen')

plt.xlabel(r'$U \: / \: V$')
plt.ylabel(r'$log(I) \: / \: nA$')

plt.xlim(np.min(Uk)-0.15, np.max(Uk)+0.15)
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.savefig('Daten/gegenfeld.pdf')

print(ufloat(params[0],errors[0]), ufloat(params[1],errors[1])) 

e = 1.60218 *1e-19
k = 1.306 * 1e-23

print(f' T = {- e/(k*params[0])}')


print (f'\n -----------------------------------Aufgabe d-----------------------------------')

I, U = np.genfromtxt('Daten/temperatur.txt', unpack = True)
N_wl = 0.95

sigma = 5.7 * 1e-12
f = 0.32
nu = 0.28

T = ((I * U - N_wl)/(f*sigma*nu))**(1/4)

print(T)

print (f'\n -----------------------------------Aufgabe e-----------------------------------')

Is = np.array([0.27, 0.6, 1.2, 2.3, 3.5]) 
Is *= 10**(-3)

h = 6.626 * 1e-34
m = 9.109 * 1e-31
q = 1.602 * 1e-19

e0 = -1 * k * T * np.log(Is * h**3 / (f * 4 * np.pi * m * q * k**2 * T**2))

print(e0)
print('in J')

e0 /= 1.60218e-19

print(e0)
print('in eV')

av = np.mean(e0)
fehler = np.std(e0, ddof=1) / np.sqrt(len(e0))

print(ufloat(av, fehler))
