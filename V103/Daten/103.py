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


#Nur Bestimmung des Elastizitätsmodul durch Biegungsmessung

print (f'\n -----------------------------------Aufgabe a-----------------------------------')

X_ms, D_ms = np.genfromtxt('Messingsquare.txt', unpack = True)
X_mr, D_mr = np.genfromtxt('Messinground.txt', unpack = True)
X_cr, D_cr = np.genfromtxt('Copperround.txt', unpack = True)
X_as, D_as = np.genfromtxt('Aluminiumsquare.txt', unpack = True)

#Einheiten anpassen
X_ms *= 1e-2 #cm -> m
X_mr *= 1e-2 
X_cr *= 1e-2
X_as *= 1e-2

D_ms *= 1e-3 #um -> m
D_mr *= 1e-3
D_cr *= 1e-3
D_as *= 1e-3
#Einseitig eingehängt Datensatz anpassen, 10 Messwerte aufgenommen
X_ms1 = X_ms[0:10]
X_mr1 = X_mr[0:10]
X_cr1 = X_cr[0:10]
X_as1 = X_as[0:10]

D_ms1 = D_ms[0:10]
D_mr1 = D_mr[0:10]
D_cr1 = D_cr[0:10]
D_as1 = D_as[0:10]

#Beidseitig eingehängt Datensatz anpassen
X_ms2 = X_ms[10:]
X_mr2 = X_mr[10:]
X_cr2 = X_cr[10:]
X_as2 = X_as[10:]

D_ms2 = D_ms[10:]
D_mr2 = D_mr[10:]
D_cr2 = D_cr[10:]
D_as2 = D_as[10:]

#Aufteilen in rechts und linksseitig
X_ms2_r = X_ms[10:15]
X_ms2_l = X_ms[15:]  
D_ms2_r = D_ms[10:15]
D_ms2_l = D_ms[15:]  

X_mr2_r = X_mr[10:15]
X_mr2_l = X_mr[15:]  
D_mr2_r = D_mr[10:15]
D_mr2_l = D_mr[15:]  

X_cr2_r = X_cr[10:15]
X_cr2_l = X_cr[15:]  
D_cr2_r = D_cr[10:15]
D_cr2_l = D_cr[15:]  

X_as2_r = X_as[10:15]
X_as2_l = X_as[15:]  
D_as2_r = D_as[10:15]
D_as2_l = D_as[15:]  

print(X_ms2_r,X_ms2_l)

def LX2(x,l): #einseitig
    return l*x**2 - x**3/3

def LX(x): #beidseitig
    l = 0.55 #m 
    return 3*x*l**2 - 4 * x**3


params_ms1, covariance_matrix = np.polyfit(LX2(X_ms1,0.548), D_ms1, deg = 1, cov = True)
uncertainties_ms1 = np.sqrt(np.diag(covariance_matrix))

params_mr1, covariance_matrix = np.polyfit(LX2(X_mr1,0.543), D_mr1, deg = 1, cov = True)
uncertainties_mr1 = np.sqrt(np.diag(covariance_matrix))

params_cr1, covariance_matrix = np.polyfit(LX2(X_cr1,0.546), D_cr1, deg = 1, cov = True)
uncertainties_cr1 = np.sqrt(np.diag(covariance_matrix))

params_as1, covariance_matrix = np.polyfit(LX2(X_as1,0.534), D_as1, deg = 1, cov = True)
uncertainties_as1 = np.sqrt(np.diag(covariance_matrix))


x = np.linspace(0, 0.1, 51)
plt.figure()
plt.plot(LX2(X_ms1,0.548) , D_ms1, '.', label = 'Messwerte ME')
plt.plot(x , params_ms1[0]*x+params_ms1[1], label = 'Ausgleichsgerade ME')

plt.xlabel(r'$(Lx^2 - x^3/3)  \, in \, m^3$')
plt.ylabel(r'$ D(x)  \, in \, m$')

plt.savefig('MS1.png')
#plt.close()

plt.plot(LX2(X_mr1,0.543) , D_mr1, '.', label = 'Messwerte MR')
plt.plot(x , params_mr1[0]*x+params_mr1[1], label = 'Ausgleichsgerade ME')

plt.savefig('MR1.png')
#plt.close()

plt.plot(LX2(X_cr1,0.546) , D_cr1, '.', label = 'Messwerte CR')
plt.plot(x , params_cr1[0]*x+params_cr1[1], label = 'Ausgleichsgerade CR')

plt.savefig('CR1.png')
#plt.close()

plt.plot(LX2(X_as1,0.534) , D_as1, '.', label = 'Messwerte AS')
plt.plot(x , params_as1[0]*x+params_as1[1], label = 'Messwerte AS')

plt.legend()
plt.savefig('AS1.png')
plt.close()

print (f'\n -----------------------------------Einseitig eingespannt-----------------------------------')
print(f'Messing eckig:{params_ms1,uncertainties_ms1}')
print(f'Messing rund:{params_mr1,uncertainties_mr1}')
print(f'Kupfer rund:{params_cr1,uncertainties_cr1}')
print(f'Alu eckig:{params_as1,uncertainties_as1}')


a_ms1 = ufloat(params_ms1[0], uncertainties_ms1[0])
a_mr1 = ufloat(params_mr1[0], uncertainties_mr1[0])
a_cr1 = ufloat(params_cr1[0], uncertainties_cr1[0])
a_as1 = ufloat(params_as1[0], uncertainties_as1[0])


print(f'Messing eckig:{747.2 * 1e-3 * 9.81/ (2*a_ms1 * 1 * 1.2 /12)}')
print(f'Alu eckig:{550 * 1e-3* 9.81/ (2*a_as1 * 1 /12)}')
print(f'Kupfer rund:{750* 1e-3 * 9.81/ (2*a_cr1 * np.pi * (1)**4 /64)}')
print(f'Messing rund:{750* 1e-3 * 9.81/ (2*a_mr1 * np.pi *(1)**4 /64)}')





params_ms2_r, covariance_matrix = np.polyfit(LX(X_ms2_r), D_ms2_r, deg = 1, cov = True)
uncertainties_ms2_r = np.sqrt(np.diag(covariance_matrix))

params_mr2_r, covariance_matrix = np.polyfit(LX(X_mr2_r), D_mr2_r, deg = 1, cov = True)
uncertainties_mr2_r = np.sqrt(np.diag(covariance_matrix))

params_cr2_r, covariance_matrix = np.polyfit(LX(X_cr2_r), D_cr2_r, deg = 1, cov = True)
uncertainties_cr2_r = np.sqrt(np.diag(covariance_matrix))

params_as2_r, covariance_matrix = np.polyfit(LX(X_as2_r), D_as2_r, deg = 1, cov = True)
uncertainties_as2_r = np.sqrt(np.diag(covariance_matrix))

x = np.linspace(-0.05, 0.15, 51)
plt.figure()
plt.plot(LX(X_ms2_r) , D_ms2_r, '.', label = 'Messwerte ME')
plt.plot(x , params_ms2_r[0]*x+params_ms2_r[1], label = 'Ausgleichsgerade ME')

plt.plot(LX(X_mr2_r) , D_mr2_r, '.', label = 'Messwerte MR')
plt.plot(x , params_mr2_r[0]*x+params_mr2_r[1], label = 'Ausgleichsgerade ME')

plt.plot(LX(X_cr2_r) , D_cr2_r, '.', label = 'Messwerte CR')
plt.plot(x , params_cr2_r[0]*x+params_cr2_r[1], label = 'Ausgleichsgerade CR')

plt.plot(LX(X_as2_r) , D_as2_r, '.', label = 'Messwerte AS')
plt.plot(x , params_as2_r[0]*x+params_as2_r[1], label = 'Messwerte AS')

plt.xlabel(r'$(3xL^2 - 4x^3)  \, in \, m^3$')
plt.ylabel(r'$ D(x)  \, in \, m$')

plt.legend()
plt.savefig('AS2_r.png')
plt.close()

print (f'\n -----------------------------------Beidseitig eingespannt - rechts betrachtet-----------------------------------')
print(f'Messing eckig :{params_ms2_r,uncertainties_ms2_r}')
print(f'Messing rund :{params_mr2_r,uncertainties_mr2_r}')
print(f'Kupfer rund :{params_cr2_r,uncertainties_cr2_r}')
print(f'Alu eckig :{params_as2_r,uncertainties_as2_r}')

a_ms2_r = ufloat(params_ms2_r[0], uncertainties_ms2_r[0])
a_mr2_r = ufloat(params_mr2_r[0], uncertainties_mr2_r[0])
a_cr2_r = ufloat(params_cr2_r[0], uncertainties_cr2_r[0])
a_as2_r = ufloat(params_as2_r[0], uncertainties_as2_r[0])


print(f'Messing eckig:{ 1238.6 * 1e-3 * 9.81/ (48*a_ms2_r * 1 * 1.2 /12)}')
print(f'Alu eckig:{747.2 * 1e-3* 9.81/ (48*a_as2_r * 1 /12)}')
print(f'Kupfer rund:{747.2* 1e-3 * 9.81/ (48*a_cr2_r * np.pi * (1)**4 /64)}')
print(f'Messing rund:{1238.6* 1e-3 * 9.81/ (48*a_mr2_r * np.pi *(1)**4 /64)}')



params_ms2_l, covariance_matrix = np.polyfit(LX(X_ms2_l), D_ms2_l, deg = 1, cov = True)
uncertainties_ms2_l = np.sqrt(np.diag(covariance_matrix))

params_mr2_l, covariance_matrix = np.polyfit(LX(X_mr2_l), D_mr2_l, deg = 1, cov = True)
uncertainties_mr2_l = np.sqrt(np.diag(covariance_matrix))

params_cr2_l, covariance_matrix = np.polyfit(LX(X_cr2_l), D_cr2_l, deg = 1, cov = True)
uncertainties_cr2_l = np.sqrt(np.diag(covariance_matrix))

params_as2_l, covariance_matrix = np.polyfit(LX(X_as2_l), D_as2_l, deg = 1, cov = True)
uncertainties_as2_l = np.sqrt(np.diag(covariance_matrix))

x = np.linspace(0.04, 0.17, 51)
plt.figure()
plt.plot(LX(X_ms2_l) , D_ms2_l, '.', label = 'Messwerte ME')
plt.plot(x , params_ms2_l[0]*x+params_ms2_l[1], label = 'Ausgleichsgerade ME')

plt.plot(LX(X_mr2_l) , D_mr2_l, '.', label = 'Messwerte MR')
plt.plot(x , params_mr2_l[0]*x+params_mr2_l[1], label = 'Ausgleichsgerade ME')

plt.plot(LX(X_cr2_l) , D_cr2_l, '.', label = 'Messwerte CR')
plt.plot(x , params_cr2_l[0]*x+params_cr2_l[1], label = 'Ausgleichsgerade CR')

plt.plot(LX(X_as2_l) , D_as2_l, '.', label = 'Messwerte AS')
plt.plot(x , params_as2_l[0]*x+params_as2_l[1], label = 'Messwerte AS')

plt.xlabel(r'$(3xL^2 - 4x^3) \, in \, m^3$')
plt.ylabel(r'$ D(x)  \, in \, m$')

plt.legend()
plt.savefig('AS2_l.png')
plt.close()

print (f'\n -----------------------------------Beidseitig eingespannt - links betrachtet-----------------------------------')
print(f'Messing eckig:{params_ms2_l,uncertainties_ms2_l}')
print(f'Messing rund :{params_mr2_l,uncertainties_mr2_l}')
print(f'Kupfer rund:{params_cr2_l,uncertainties_cr2_l}')
print(f'Alu eckig:{params_as2_l,uncertainties_as2_l}')



