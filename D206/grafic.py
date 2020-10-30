# ' ' = " "
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import sympy
from scipy.optimize import curve_fit


plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 16

t, T_1, p_1, T_2, p_2, N = np.genfromtxt('daten.txt', unpack=True)

p_1 += 1 #Auf die Drücke muss jeweils 1 bar addiert werden (siehe Aufgabenstellung)
p_2 += 1

t *= 60 #min in s

T_1 += 273.15 # Von Ceclsius in Kelvin
T_2 += 273.15

def f(x,A,B,C): #Funktion an die gefittet wird
    return A*x**2 + B*x + C

parameter1, covariance_matrix_1 = curve_fit(f,t,T_1) #Aufruf der Fitfunktion
parameter2, covariance_matrix_2 = curve_fit(f,t,T_2)

for names, value in zip('ABC', parameter1): #Ausgabe der Werte
    print(f"{names}={value}")

for names, value in zip('ABC', parameter2): #Ausgabe der Werte
    print(f"{names}={value}")



#plt.plot(t,T_1,'.', label=r"$Eimer 1$")
plt.xlabel(r'$t \, [s]$')
plt.ylabel(r'$T  \, [K]$')

plt.plot(t, f(t, *parameter1), label=r'$Fit \, T_1(t)$')
plt.errorbar(t, T_1, xerr=0, yerr=0.1, fmt='.', label = r'$T_1(t)$')

plt.plot(t, f(t, *parameter2), label=r'$Fit \, T_2(t)$')
plt.errorbar(t, T_2, xerr=0, yerr=.1, fmt='.', label = r'$T_2(t)$')

plt.legend()
plt.tight_layout()
plt.savefig('grafic.pdf')


# c) 4 Temperaturen t=8, 16, 24, 32  

x = sympy.var('x')
T_1f = f(x, *parameter1)
T_2f = f(x, *parameter2)

T_1fdif1 = T_1f.diff(x)
T_2fdif1 = T_2f.diff(x)

