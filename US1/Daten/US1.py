import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import scipy.constants as const
from uncertainties import ufloat
from scipy import optimize
from uncertainties.umath import *
from scipy.optimize import fsolve

d, t, A=np.genfromtxt('a.txt', unpack = True)
d*=2 #doppelte Strecke 
params, covariance_matrix = np.polyfit(t, d, deg=1, cov=True)
x_plot = np.linspace(10.5, 46.7)

plt.plot(t, d,'kx',label = 'Messdaten')
plt.plot(x_plot, params[0] * x_plot + params[1], 'g-', label='Lineare Regression', linewidth=1.5)
plt.ylabel(r'$2 \cdot d \,/\, \mathrm{mm}$')
plt.xlabel(r'$t \,/\, \mathrm{\mu s}$')
plt.legend()
plt.tight_layout()
plt.savefig("schall.pdf")
plt.close() 


def exp(x,I0,a):
    return I0*np.exp(-x * a)

    
p0=[9.2, 0.00075]

param, cov = curve_fit(exp, d, A, p0 =p0)

plt.figure()
plt.plot(d, A, 'kx', label = "Messwerte")
xplot = np.linspace(d[2],d[8],1000)
plt.plot(xplot,exp(xplot,*param),'g', label = "Ausgleichsfunktion")
plt.xlabel(r'$d \, / \, \mathrm{mm} $')
plt.ylabel(r'$I \, / \, \mathrm{V}$')
plt.legend()
plt.tight_layout()
plt.savefig("daempf.pdf")
plt.close() 

print(f"schallgeschwindigkeit c = {params[0]}")
print(param)