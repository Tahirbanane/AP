# ' ' = " "
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import sympy
from scipy.optimize import curve_fit
from uncertainties import ufloat


plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 16

t, T_1, p_1, T_2, p_2, N = np.genfromtxt('daten.txt', unpack=True)

p_1 += 1 #Auf die Drücke muss jeweils 1 bar addiert werden (siehe Aufgabenstellung)
p_2 += 1

t *= 60 #min in s

T_1 += 273.15 # Von Ceclsius in Kelvin
T_2 += 273.15

print(f"\n \n \n Aufgabe b):")

def f(x,A,B,C): #Funktion an die gefittet wird
    return A*x**2 + B*x + C

parameter1, covariance_matrix_1 = curve_fit(f,t,T_1) #Aufruf der Fitfunktion
parameter2, covariance_matrix_2 = curve_fit(f,t,T_2)

uncertainties1 = np.sqrt(np.diag(covariance_matrix_1))
uncertainties2 = np.sqrt(np.diag(covariance_matrix_2))

for names ,value, uncertaintie in zip("ABC", parameter1,uncertainties1):
    names = ufloat(value, uncertaintie)
    print(f"Die ersten drei Werte für T_1 mit Fehler sind: \n {names:.8f}")
for names ,value, uncertaintie in zip("DEF", parameter2,uncertainties2):
    names = ufloat(value, uncertaintie)
    print(f"Die ersten drei Werte  für T_2 mit Fehler sind: \n {names:.8f}")


# c) 4 Temperaturen t= 8, 16, 24, 32  

print(f"\n \n \n Aufgabe c):")

x = sympy.var('x')
T_1f = f(x, *parameter1)
T_2f = f(x, *parameter2)

T_1f_1 = T_1f.diff(x)
T_2f_1 = T_2f.diff(x)

cash = np.array([0., 0., 0., 0., 0., 0., 0., 0.]) #Array mit den Ergebnissen des einsetzens 0-3 T_1f_1 und 4-7 T_2f_1
#weil ich nicht herausbekkomen hab wie man zahlen einsetzt die quick and unrelayable version
#Für dT_1/dt gilt:
for value in range(1,5):
    cash[value - 1] = parameter1[0] + parameter1[1] * (8*value)
    print(f"Das Ergebnis für T_1f_1(t={8*value}) = {cash[value - 1]} ")

#Für dT_2/dt gilt:
for value in range(1,5):
    cash[value+3] = parameter2[0] + parameter2[1] * (8*value)
    print(f"Das Ergebnis für T_2f_1(t={8*value}) = {cash[value+3]} ")



print(f"\n \n \n Aufgabe d):")

#Temperatur bei t=8 , T_1 =  ;T_2 =  
#Temperatur bei t=16, T_1 =  ;T_2 = 
#Temperatur bei t=24, T_1 =  ;T_2 = 
#Temperatur bei t=32, T_1 =  ;T_2 = 





#Grafik
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