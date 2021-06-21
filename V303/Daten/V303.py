import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import scipy.constants as const
from uncertainties import ufloat
from scipy import optimize
from uncertainties.umath import *
from scipy.optimize import fsolve

print(f"------------------------phase------------------------")

alpha, U_o, U_m = np.genfromtxt('phase.txt', unpack=True)

def radians(theta):
    return theta/180 * np.pi

def cos(phi, a, b, c, d):
    return a*np.cos(b*phi+c) + d

alpha = radians(alpha)

params1, pcov1 = curve_fit(cos, alpha, U_o)
params2, pcov2 = curve_fit(cos, alpha, U_m)

plt.figure()
plt.plot(alpha, U_o, 'kx', label='Messdaten')
x = np.linspace(0 , np.pi ,100)
plt.plot(x, cos(x, *params1), color='steelblue',ls='-', label=r'$cos(\phi)$')
plt.ylabel(r'$U_o \,/\, \mathrm{V}$')
plt.xlabel(r'$\phi$')
plt.legend()
plt.xticks([0, np.pi/3, 2*np.pi/3, np.pi], [r"$0$", r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", r"$\pi$"])
plt.tight_layout()
plt.savefig('phase.pdf')

print(f"U_ohne params: a = {params1[0]}, b = {params1[1]}, c = {params1[2]}, d = {params1[3]}")

plt.figure()
plt.plot(alpha, U_m, 'kx', label='Messdaten')
x = np.linspace(0 , np.pi ,100)
plt.plot(x, cos(x, *params2), color='steelblue',ls='-', label=r'$cos(\phi)$')
plt.ylabel(r'$U_m \,/\, \mathrm{V}$')
plt.xlabel(r'$\phi$')
plt.legend()
plt.xticks([0, np.pi/3, 2*np.pi/3, np.pi], [r"$0$", r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", r"$\pi$"])
plt.tight_layout()
plt.savefig('phasenoise.pdf')

print(f"U_mit params: a = {params2[0]}, b = {params2[1]}, c = {params2[2]}, d = {params2[3]}")


print(f"-------------------------led-------------------------")
d, U = np.genfromtxt('led.txt', unpack=True)

d -= 5 #nullkorrektur
d /=100 #m 

def U_fit1(r,a,b):
    return a/r + b

def U_fit2(r,a,b):
    return a/(r**2) + b

params1, pcov1 = curve_fit(U_fit1,d,U)
params2, pcov2 = curve_fit(U_fit2,d,U)

plt.figure()
plt.plot(d, U, 'kx', label='Messdaten')
plt.ylabel(r'$U \,/\, \mathrm{V}$')
plt.xlabel(r'$d \,/\, \mathrm{m}$')
x = np.linspace(0.0455 , 0.37 ,1000)
plt.plot(x, U_fit1(x, *params1), color='steelblue',ls='-', label=r'$\mathrm{\frac{1}{r}}$')
plt.plot(x, U_fit2(x, *params2), color='sandybrown', ls = '-', label=r'$\mathrm{\frac{1}{r^2}}$')
plt.legend()
plt.tight_layout()
plt.savefig('led.pdf')

print(f"r1 fit params: a = {params1[0]}, b = {params1[1]}")
print(f"r2 fit params: a = {params2[0]}, b = {params2[1]}")


