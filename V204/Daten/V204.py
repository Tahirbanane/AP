import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import sympy
from scipy.optimize import curve_fit
from uncertainties import ufloat


t_1, T_1_1, T_4_1, T_5_1, T_8_1, T_2_1, T_3_1, T_6_1, T_7_1 = np.genfromtxt('Daten/statisch.txt', unpack=True)
t_2, T_1_2, T_2_2, T_3_2, T_4_2, T_5_2, T_6_2, T_7_2, T_8_2 = np.genfromtxt('Daten/dynamisch_80s.txt', unpack=True)
t_3, T_1_3, T_2_3, T_3_3, T_4_3, T_5_3, T_6_3, T_7_3, T_8_3 = np.genfromtxt('Daten/dynamisch_200s.txt', unpack=True)

t_1 /= 10 #auf sekunden
t_2 *= 2
t_3 *= 2


#statische Methode Plot:
plt.figure()
plt.xlabel(r'$t \, [s]$')
plt.ylabel(r'$T  \, [K]$')

plt.plot(t_1, T_1_1, label=r'$T_1$')
plt.plot(t_1, T_4_1, label=r'$T_4$')
plt.plot(t_1, T_5_1, label=r'$T_5$')
plt.plot(t_1, T_8_1, label=r'$T_8$')

plt.legend()
plt.tight_layout()
plt.savefig('grafic.pdf')

#statische Methode 700s:
print(f'Die Temperatur bei T1 beträgt: {T_1_1[7000]}, \n Die Temperatur bei T4 beträgt: {T_4_1[7000]}, \n Die Temperatur bei T5 beträgt: {T_5_1[7000]}, \n Die Temperatur bei T8 beträgt: {T_1_1[7000]}, \n')

#Wärmestrom:
def kappa(roh,c, x, t, A_nah, A_fern):
    return (roh*c*x**2)/(2*t*np.log(A_nah/A_fern)) #hier ist A die Amplitude

T_12_1 = T_1_1 - T_2_1

#Amplitudenberechnung:
#def A:



for i in range(1,5):
    print(f'Der Wärmestrom zum Zeitpunkt t={i*100} beträgt {-kappa(8520, 385, 0.03, 80, T_1_1[i*1000], T_2_1[i*1000])*0.012*0.004*T_12_1[i*1000]} ')



