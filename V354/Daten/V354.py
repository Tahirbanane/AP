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

t, U = np.genfromtxt('a.txt', unpack = True)

t *= 10**(-6) #Zeit wurde in Mikrosekunden gemessen
U1 = abs(U) # da nur die Amplitude unterscuht wird, schauen wir uns die einhüllende an
U2 = np.array([U[1], U[3], U[5], U[7], U[9], U[11], U[13], U[15]])
t2 = np.array([t[1], t[3], t[5], t[7], t[9], t[11], t[13], t[15]])

def f(x, a, b):
    return a*np.exp(b*x) # b = -2*np.pi*μ

params1, covarianzmatrix1 = curve_fit(f, t, U1)
error1 = np.sqrt(np.diag(covarianzmatrix1))

params2, covarianzmatrix2 = curve_fit(f, t2, U2)
error2 = np.sqrt(np.diag(covarianzmatrix2))


plt.figure()
plt.plot(t, U, 'x',label = 'Messwerte')
plt.plot(t, f(t, *params1), label = 'Ausgleichskurve1')
plt.plot(t2, f(t2, *params2), label = 'Ausgleichskurve2')

plt.xlabel(r'$t \: / \: s$')
plt.ylabel(r'$U_C \: / \: V$')

plt.tight_layout()
plt.legend()
#plt.show()
plt.savefig('a.jpg')

print('a1 = {:.10f} ± {:.10f}'.format(params1[0], error1[0]))
print('b1 = {:.4f} ±   {:.5f}'.format(params1[1], error1[1]))

print('a2 = {:.10f} ± {:.10f}'.format(params2[0], error2[0]))
print('b2 = {:.4f} ±   {:.5f}'.format(params2[1], error2[1]))

print('T_ex = {:.10f} ± {:.10f}'.format(1/params1[0], np.sqrt(1/(params1[0]**2/2*np.pi) * error1[0]**2)))


print (f'\n -----------------------------------Aufgabe b-----------------------------------')

# diskrepanz da Drähte kabel und co ebenfalls einen wiederstand haben, theorie wert sollte niedriger als der experimentell festgestellte wert sein

R = np.genfromtxt('b.txt', unpack = True)

L = 3.5 * 10*(-3) #Henry