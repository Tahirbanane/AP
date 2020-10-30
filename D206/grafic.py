import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 16

t, T_1, p_1, T_2, p_2, N = np.genfromtxt('daten.txt', unpack=True)

p_1 += 1 #Auf die Drücke muss jeweils 1 bar addiert werden (siehe Aufgabenstellung)
p_2 += 1

T_1 += 273.15 # Von Ceclsius in Kelvin
T_2 += 273.15

def f(x,A,B,C): #Funktion an die gefittet wird
    return A*x**2 + B*x + C

parameter1, covariance_matrix = curve_fit(f,t,T_1) #Aufruf der Fitfunktion

for names, value in zip('ABC', parameter1): #Ausgabe der Werte
    print(f"{names}={value}")


#plt.plot(t,T_1,'.', label=r"$Eimer 1$")
plt.xlabel('t / s')
plt.ylabel(r'$T_1 / \, °C$')
plt.errorbar(t, T_1, xerr=0, yerr=0.1)

#plt.legend()
plt.tight_layout()
plt.savefig('grafic.pdf')

# ' ' = " "