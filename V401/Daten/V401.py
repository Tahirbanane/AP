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
d1,N1=np.genfromtxt('a.txt', unpack = True)
d1 = d1/1000      #in meter
d1 = d1/5.046     #mit übersetzung
lambda1 = 2*d1/N1 
lambda1_mittel=sum(lambda1)/10
lambda_std = np.std(lambda1, ddof=1)

print(f"berechnete Wellenlänge: {lambda1}")
print(f"Mittelwert: {lambda1_mittel}")
print(f"Standardabweichung: {lambda_std}")

print (f'\n -----------------------------------Aufgabe b-----------------------------------')
b = 0.05 #m 
T_0 = 273.15 #K
T = 293.15 #K
p_0 = 1.0132 #bar
dp,N2=np.genfromtxt('b.txt', unpack = True)
dp = dp*0.00133322 #torr in bar
n = 1 + ((N2 * lambda1_mittel * T * p_0)/(2 * b * T_0 * dp))
n_mittel = np.mean(n)
n_std = np.std(n, ddof=1)

print(f"Brechungsindex: {n}")
print(f"Mittelwert: {n_mittel}")
print(f"Standardabweichung: {n_std}")
