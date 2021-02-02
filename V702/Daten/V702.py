#import matplotlib.pyplot as plt
#import numpy as np
#import uncertainties.unumpy as unp
#import sympy
#import statistics
#from scipy.optimize import curve_fit
#from uncertainties import unumpy, ufloat
#from scipy.signal import find_peaks
#import scipy.constants as const
#import pandas as pd
#from uncertainties.unumpy import (nominal_values as noms,
#                                  std_devs as stds)
#
#
#print (f' -----------------------------------Untergrund-----------------------------------')
#
#N = np.genfromtxt('untergrund.txt', unpack = True)
#
#N_U = sum(N) / len(N)
#dN_U = np.sqrt(N_U)
#
#print(f'durchschnittliche Untergrundrate: {N_U} +- {dN_U}')
#
#print (f'\n -----------------------------------Vanadium-----------------------------------')
#
#t, N = np.genfromtxt('Vanadium.dat', unpack = True)
#
#N -= N_U/10 #untergrund abziehen
#dN = np.sqrt(N) #poisson
#
#params, covariance_matrix = np.polyfit(t, np.log(N), 1, cov=True)
#err = np.sqrt(np.diag(covariance_matrix))
#
#print(f'Zerfallskonstante: {params[0]} +- {err[0]}')
#print(f'Achsenabschnitt: {params[1]} +- {err[1]}')
#
#x = np.linspace(t[0], t[-1], 1000)
#
#plt.figure()
#plt.errorbar(t, N, xerr = 0, yerr = dN, fmt = '.', label = 'Messpunkte')
#plt.plot(x, 10**2 * 2 * np.exp(params[0] * x),
#        color='r',
#        label=f'Fit')
#
#plt.fill_between(x, 10**2 * 2 * np.exp(-1 * (-params[0]-err[0]) * x),
#         10**2 * 2 * np.exp(-1 * (-params[0]+err[0]) * x),
#         alpha=0.2, color='r')
#
#plt.xscale('log')
#plt.xlabel(r't in s')
#plt.ylabel(r'Zerfälle / 30s')
#plt.legend()
##plt.show()
#plt.savefig('Vanadium.pdf')
#
#print(f'Halbwertszeit von Vanadium: {np.log(2) / ufloat(-params[0], err[0])}')
##print(N)
#N = N[0:10]
#t = t[0:10]
#params, covariance_matrix = np.polyfit(t, np.log(N), 1, cov=True)
#err = np.sqrt(np.diag(covariance_matrix))
#
#print(f'\nZerfallskonstante: {params[0]} +- {err[0]}')
#print(f'Achsenabschnitt: {params[1]} +- {err[1]}')
#
#print(f'Halbwertszeit von Vanadium genauer: {np.log(2) / ufloat(-params[0], err[0])}')
#
#
#print (f'\n -----------------------------------Rhodium-----------------------------------')
#
#t, N = np.genfromtxt('Rhodium.dat', unpack = True)
#
#t, N = np.genfromtxt('Vanadium.dat', unpack = True)
#
#N -= N_U/20 #untergrund abziehen
#dN = np.sqrt(N) #poisson
#
#plt.figure()
#plt.errorbar(t, N, xerr = 0, yerr = dN, fmt = '.', label = 'Messpunkte')
##plt.xscale('log')
#plt.xlabel(r't in s')
#plt.ylabel(r'Zerfälle / 15s')
#
#
##Berechnung für den langsamen Zerfall
#Nl = N[10:20]
#tl = t[10:20]
#print(f'\nlangsamer Zerfall ab:{tl[0]}')
#
#params, covariance_matrix = np.polyfit(tl, np.log(Nl), 1, cov=True)
#err = np.sqrt(np.diag(covariance_matrix))
#
#print(f'Zerfallskonstante: {params[0]} +- {err[0]}')
#print(f'Achsenabschnitt: {params[1]} +- {err[1]}')
#
#x = np.linspace(t[0], t[-1], 1000)
#plt.plot(x, 10**2 * 2 * np.exp(params[0] * x),
#        color='r',
#        label=f'Fit langsamer Zerfall')
#
#plt.fill_between(x, 10**2 * 2 * np.exp(-1 * (-params[0]-err[0]) * x),
#         10**2 * 2 * np.exp(-1 * (-params[0]+err[0]) * x),
#         alpha=0.2, color='r')
#
#print(f'Halbwertszeit von Rhodium lang: {np.log(2) / ufloat(-params[0], err[0])}')
#
#plt.legend()
##plt.show()
#plt.savefig('Rhodium.pdf')
#
##Berechnung für den schnellen Zerfall
#
#Nk = N[0:10]
#tk = t[0:10]
#
#print(f'\nschneller Zerfall ab:{tk[0]}')
#
#N_korr = 667 * (1 - np.exp(params[0] * 15)) * np.exp(params[0] * tk)
#params, covariance_matrix = np.polyfit(tk, np.log(Nk - N_korr), 1, cov=True)
#err = np.sqrt(np.diag(covariance_matrix))
#print(f'Halbwertszeit von Rhodium schnell korrigiert: {np.log(2) / ufloat(-params[0], err[0])}')
#print(f'Steigung der Ausgleichsgerade: {-params[0]} +- {err[0]} und {params[1]} +- {err[1]} ')
#
#
#x = np.linspace(t[0], t[-1], 1000)
#plt.plot(x, 10**2 * 2 * np.exp(params[0] * x),
#        color='b',
#        label=f'Fit schneller Zerfall')
#
#plt.fill_between(x, 10**2 * 2 * np.exp(-1 * (-params[0]-err[0]) * x),
#         10**2 * 2 * np.exp(-1 * (-params[0]+err[0]) * x),
#         alpha=0.1, color='b')
#        
#print(f'Halbwertszeit von Rhodium kurz: {np.log(2) / ufloat(-params[0], err[0])}')
#
#plt.legend()
##plt.show()
#plt.savefig('Rhodium2.pdf')