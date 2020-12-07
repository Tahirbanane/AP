import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import sympy
from scipy.optimize import curve_fit
from uncertainties import ufloat
from scipy.signal import find_peaks


t_1, T_1_1, T_4_1, T_5_1, T_8_1, T_2_1, T_3_1, T_6_1, T_7_1 = np.genfromtxt('statisch.txt', unpack=True)
t_2, T_1_2, T_2_2, T_3_2, T_4_2, T_5_2, T_6_2, T_7_2, T_8_2 = np.genfromtxt('dynamisch_80s.txt', unpack=True)
t_3, T_1_3, T_2_3, T_3_3, T_4_3, T_5_3, T_6_3, T_7_3, T_8_3 = np.genfromtxt('dynamisch_200s.txt', unpack=True)

t_1 /= 10 #auf sekunden
t_2 *= 2
t_3 *= 2

T_1_1 += 273.15
T_4_1 += 273.15
T_5_1 += 273.15
T_8_1 += 273.15
T_2_1 += 273.15
T_3_1 += 273.15
T_6_1 += 273.15
T_7_1 += 273.15

T_1_2 += 273.15
T_4_2 += 273.15
T_5_2 += 273.15
T_8_2 += 273.15
T_2_2 += 273.15
T_3_2 += 273.15
T_6_2 += 273.15
T_7_2 += 273.15

T_1_3 += 273.15
T_4_3 += 273.15
T_5_3 += 273.15
T_8_3 += 273.15
T_2_3 += 273.15
T_3_3 += 273.15
T_6_3 += 273.15
T_7_3 += 273.15

x1, _ = find_peaks(T_1_2, distance = 15)
x2, _ = find_peaks(-T_1_2, distance = 15)


plt.figure()
plt.xlabel(r'$t \, [s]$')
plt.ylabel(r'$T  \, [K]$')

plt.plot(t_2, T_1_2, label=r'$T_1$')
plt.plot(t_2[x1], T_1_2[x1],'.r', label=r'$x1$')
plt.plot(t_2[x2], T_1_2[x2],'.', label=r'$x2$')


plt.legend()
plt.tight_layout()
plt.savefig('kontroll.pdf')

#Wärmestrom:
def kappa(roh,c, x, t, A_nah, A_fern):
    return (roh*c*(x**2))/(2*t*np.log(A_nah/A_fern)) #hier ist A die Amplitude

#Durchschnitt:
def ds(A):
    sum = 0
    for i in (0, len(A)-1):
        sum += A[i]
    return sum/(len(A))

#Betrag:
def betrag(A):
    for i in range(0,len(A)-1):
        if A[i] < 0:
            A[i] *= -1
    return A

def betrag_z(A):
    if(A < 0):
        return -A
    return A



