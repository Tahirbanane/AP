
import os
import pathlib
os.environ['MATPLOTLIBRC'] = (pathlib.Path(__file__).absolute().parent.parent.parent / 'default' / 'matplotlibrc').__str__()
os.environ['TEXINPUTS'] =  (pathlib.Path(__file__).absolute().parent.parent.parent / 'default').__str__() + ':'

import json

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

#plt.figure()
#plt.plot(F7,W157, '--', label = '15°')
#plt.plot(F7,W307, '--', label = '30°')
#plt.plot(F7,W607, '--', label = '60°')
#
#plt.tight_layout()
#plt.legend()
#plt.savefig('7mm.pdf')
#
#
#
#plt.figure()
#plt.plot(F10,W1510, '--', label = '15°')
#plt.plot(F10,W3010, '--', label = '30°')
#plt.plot(F10,W6010, '--', label = '60°')
#
#plt.tight_layout()
#plt.legend()
#plt.savefig('10mm.pdf')
#
#
#plt.figure()
#plt.plot(F16,W1516, '--', label = '15°')
#plt.plot(F16,W3016, '--', label = '30°')
#plt.plot(F16,W6016, '--', label = '60°')
#
#plt.tight_layout()
#plt.legend()
#plt.savefig('16mm.pdf')


print (f'\n -----------------------------------Aufgabe a-----------------------------------')

F, W15, W30, W60 = np.genfromtxt('a.txt', unpack = True)

F10 = F[0:6] #10mm Durchmesser
W1510 = W15[0:6]
W3010 = W30[0:6]
W6010 = W60[0:6]

F16 = F[6:12] #16mm Durchmesser
W1516 = W15[6:12]
W3016 = W30[6:12]
W6016 = W60[6:12]

F7 = F[12:] #16mm Durchmesser
W157 = W15[12:]
W307 = W30[12:]
W607 = W60[12:]

#Konstanten
cL = 1800   #meter pro sekunde
cP = 2700   #meter pro sekunde
f = 2000000 #hertz

def gerade(a,b):
    return a*x+b

def alpha(theta):
    return np.pi/2 - np.arcsin(np.sin(np.deg2rad(theta)) * cL/cP)

def v(dF, alpha):
    return (dF * cL)/(2 * f * np.cos(alpha)) 
    

print(v(W15, alpha(15)),v(W30, alpha(30)), v(W60, alpha(60)))

Ergebnisse = json.load(open('Ergebnisse.json','r'))
if not 'Winkel' in Ergebnisse:
    Ergebnisse['Winkel'] = {}
Ergebnisse['Winkel']['alpha1'] = alpha(15)
Ergebnisse['Winkel']['alpha2'] = alpha(30)
Ergebnisse['Winkel']['alpha3'] = alpha(60)
Ergebnisse['WinkelGrad']['alpha1'] = np.rad2deg(alpha(15))
Ergebnisse['WinkelGrad']['alpha2'] = np.rad2deg(alpha(30))
Ergebnisse['WinkelGrad']['alpha3'] = np.rad2deg(alpha(60))
json.dump(Ergebnisse,open('Ergebnisse.json','w'),indent=4)


plt.figure()
plt.plot(v(W1516, alpha(15)), W1516/np.cos(alpha(15)), '.', label='Messwerte beim Winkel 15°')
plt.plot(v(W3016, alpha(30)), W3016/np.cos(alpha(30)), '.', label='Messwerte beim Winkel 30°')
plt.plot(v(W6016, alpha(60)), W6016/np.cos(alpha(60)), '.', label='Messwerte beim Winkel 60°')

plt.xlabel(r'Strömungsgeschwindigkeit [$\frac{m}{s}$]')
plt.ylabel(r'$\frac{Δv}{cos(\alpha)}$')

plt.legend(loc='best')
plt.tight_layout()
plt.grid()
plt.savefig('16mm.pdf')

plt.figure()
plt.plot(v(W1510, alpha(15)), W1510/np.cos(alpha(15)), '.', label='Messwerte beim Winkel 15°')
plt.plot(v(W3010, alpha(30)), W3010/np.cos(alpha(30)), '.', label='Messwerte beim Winkel 30°')
plt.plot(v(W6010, alpha(60)), W6010/np.cos(alpha(60)), '.', label='Messwerte beim Winkel 60°')

plt.xlabel(r'Strömungsgeschwindigkeit [$\frac{m}{s}$]')
plt.ylabel(r'$\frac{Δv}{cos(\alpha)}$')

plt.legend(loc='best')
plt.tight_layout()
plt.grid()
plt.savefig('10mm.pdf')

plt.figure()
plt.plot(v(W157, alpha(15)), W157/np.cos(alpha(15)), '.', label='Messwerte beim Winkel 15°')
plt.plot(v(W307, alpha(30)), W307/np.cos(alpha(30)), '.', label='Messwerte beim Winkel 30°')
plt.plot(v(W607, alpha(60)), W607/np.cos(alpha(60)), '.', label='Messwerte beim Winkel 60°')

plt.xlabel(r'Strömungsgeschwindigkeit [$\frac{m}{s}$]')
plt.ylabel(r'$\frac{Δv}{cos(\alpha)}$')

plt.legend(loc='best')
plt.tight_layout()
plt.grid()
plt.savefig('7mm.pdf')

print (f'\n -----------------------------------Aufgabe b-----------------------------------')

T, W, I = np.genfromtxt('b.txt', unpack = True)

I *= 1e-2

W = np.abs(W) #Der Betrag wird genommen

T70 = T[:10]
W70 = W[:10]
I70 = I[:10]

T45 = T[10:]
W45 = W[10:]
I45 = I[10:]

plt.figure()
plt.plot(T70,v(W70, alpha(15)),'.', label = r'Geschwindigkeit v bei 70% Leistung')
plt.plot(T70,I70,'.', label = r'Intensität I bei 70% Leistung')

plt.xlabel(r'Messtiefe [$\mu s$]')
plt.ylabel(r'I [$\frac{10MV^2}{s}] / v [\frac{m}{s}$]')

plt.legend(loc='best', fontsize = 10)
plt.tight_layout()
plt.grid()
plt.savefig('70.pdf')

plt.figure()
plt.plot(T45,v(W45, alpha(15)),'.', label = r'Geschwindigkeit v bei 45% Leistung')
plt.plot(T45,I45,'.', label = r'Intensität I bei 45% Leistung')

plt.xlabel(r'Messtiefe [$\mu s$]')
plt.ylabel(r'I $[\frac{10MV^2}{s}]$ / $v$ $[\frac{m}{s}]$')

plt.legend(loc='best')
plt.tight_layout()
plt.grid()
plt.savefig('45.pdf')

