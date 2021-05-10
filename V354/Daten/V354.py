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

t *= 10**(-3) #Zeit wurde in Mikrosekunden gemessen
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
plt.subplot(2,1,1)
plt.plot(t, U, 'x',label = 'gemesse Spannung')
plt.plot(t, f(t, *params1), label = 'Ausgleichskurve')
#plt.plot(t2, f(t2, *params2), label = 'Ausgleichskurve2')

plt.xlabel(r'$t \: / \: s$')
plt.ylabel(r'$U_C \: / \: V$')

plt.tight_layout()
plt.legend()
#plt.show()
#plt.savefig('a.jpg')

plt.subplot(2,1,2)
plt.plot(t, U1, 'gx',label = 'gemessener Spannungsbetrag')
plt.plot(t, f(t, *params1), label = 'Ausgleichskurve')

plt.xlabel(r'$t \: / \: s$')
plt.ylabel(r'$U_C \: / \: V$')

plt.tight_layout()
plt.legend()
plt.savefig('a.pdf')

print('a1 = {:.10f} ± {:.10f}'.format(params1[0], error1[0]))
print('b1 = {:.4f} ±   {:.5f}'.format(params1[1], error1[1]))
print(f'mu = { params1[1]/ (-2*np.pi)} +- {error1[1]/(-2*np.pi)} ')

#print('a2 = {:.10f} ± {:.10f}'.format(params2[0], error2[0]))
#print('b2 = {:.4f} ±   {:.5f}'.format(params2[1], error2[1]))

print('T_ex = {:.10f} ± {:.10f}'.format(1/params1[0], np.sqrt(1/(params1[0]**2/2*np.pi) * error1[0]**2)))

b = ufloat(params1[1], error1[1])
L = ufloat(3.5*1e-3, 0.01*1e-3) #Henry
Rv = ufloat(30.3,0,1)

Reff = - 2*L*b
print('Reff = ',Reff)

print('p = ',(Reff-Rv)/Rv)

print (f'\n -----------------------------------Aufgabe b-----------------------------------')

# diskrepanz da Drähte kabel und co ebenfalls einen wiederstand haben, theorie wert sollte niedriger als der experimentell festgestellte wert sein

R = np.genfromtxt('b.txt', unpack = True)

C = ufloat(999*1e-9, 0.1*1e-9) #Farad Kapazität fehlt

Rt = 2*(L/C)**(1/2)

print(f'R_ap = {Rt}')

print(f'p = {(Rt.n-R)/Rt.n}') #Messwerte überprüfen?

print (f'\n -----------------------------------Aufgabe c-----------------------------------')

f, UC, Ue = np.genfromtxt('c.txt', unpack = True)

f *= 1e3
UC *= 1e-3
Ue *= 1e-3


def k(x, a, b):
    return a*np.exp(-1/b*x**2)

params, covarianzmatrix = curve_fit(k, f, UC/Ue)
error = np.sqrt(np.diag(covarianzmatrix))



print(max(UC/Ue))

plt.figure()
plt.plot(f, UC/Ue, 'x', label = 'Messpunkte')
plt.plot(f, k(f, *params), label = 'Ausgleichskurve')
plt.axhline(y = max(UC/Ue) / np.sqrt(2), color = 'tab:red', label = r'$\frac{U_{max}}{U_{err}}\cdot \frac{1}{\sqrt{2}}$' )

plt.xlabel(r'$f \: / \: Hz$')
plt.ylabel(r'$\frac{U_C}{U} \: $')
plt.yscale('log')
#plt.xscale('log')

plt.tight_layout()
plt.legend(loc = 'center left')
plt.show()
#plt.savefig('c.jpg')

