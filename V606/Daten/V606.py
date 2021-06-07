import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import scipy.constants as const
from uncertainties import ufloat
from scipy import optimize
from uncertainties.umath import *
from scipy.optimize import fsolve


e=const.e
hbar=const.hbar
m=const.m_e
mu_0=const.mu_0
k=const.k
N_A=const.N_A

w, U_A = np.genfromtxt('a.txt', unpack=True)
U_A=U_A/10 #verstärkung mV
U=U_A/496 #mV

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

popt, _ = optimize.curve_fit(gaussian, w[4:], U[4:],p0=[30, 15, 30])
print(popt)
x = np.linspace(20.8, 42, 1000)
#plt.plot(x, data)

def h (x, x0, y, a, k, i):
    return k*x**3 + i*x**9 + a/((x**2-x0**2)**2+y**2*x0**2) 

params, covarianzmatrix = optimize.curve_fit(h, w, U, p0 = [35.5076301,1.05833979,1436.58107,0.03,0.2]) #Lorentzfunktion ist mit der Modifikation nur auf diesem Intervall die beste Approximation
print(params)
a = 1.0688491170389607

errors = np.sqrt(np.diag(covarianzmatrix))
m = ufloat(params[0], errors[0])
y = ufloat(params[1], errors[1])
alpha = ufloat(params[2], errors[2])
k = ufloat(params[3], errors[3])
i = ufloat(params[4], errors[4])
print(f'm = {m} \ny = {y} \nalpha = {alpha} \nk = {k} \ni = {i}')

nu_plus = m + (sqrt(1/(2*a) * np.log(2)))/2
nu_minus = m - (sqrt(1/(2*a) * np.log(2)))/2
print(f'𝜈_+ = {nu_plus}')
print(f'𝜈_- = {nu_minus}')
print(f'd𝜈 = {nu_plus - nu_minus}')
q=m/(nu_plus - nu_minus)
print(f"güte q = {q}")


#def f(x, x0, y, a):
#    return a/((x**2-x0**2)**2+y**2*x0**2)
#
#par, cov = optimize.curve_fit(f, U, w)
#x_plot = np.linspace(20.8, 41, 1000)
#plt.plot(x_plot, f(x_plot, *par), 'b-', label=r'Ausgleichskurve', linewidth=1)
#
#x0 = ufloat(par[0], np.sqrt(cov[0][0]))
#y = ufloat(par[1], np.sqrt(cov[1][1]))
#a = ufloat(par[2], np.sqrt(cov[2][2]))
#nu_0 = x0

plt.plot(w, U,'k.', label ='Messwerte')
plt.plot(x, h(x, *params),color='steelblue',ls='-', label = 'Lorentzkurve')
plt.plot(x, gaussian(x, *popt),color='sandybrown', ls = '-', label = 'Gaußkurve')
plt.xlabel(r'Frequenz $\nu$ / $\mathrm{kHz}$')
plt.ylabel(r'Spannungsverhältnis $\frac{U_A}{U_E}$')
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('x6.pdf')
plt.close()


plt.plot(w[13:22], U[13:22],'k.', label ='Messwerte')
print(max(h(x, *params)))
s=np.linspace(34, 37, 1000)
plt.plot(s, h(s, *params), color='steelblue',ls='-', label = 'Lorentzkurve')
plt.plot(s, 0*s + max(h(s, *params))/np.sqrt(2), 'g-',label= r'$\frac{U_{A, max}}{\sqrt{2}}$', linewidth=1.3) #Gerade
plt.xlabel(r'Frequenz $\nu$ / $\mathrm{kHz}$')
plt.ylabel(r'Spannungsverhältnis $\frac{U_A}{U_E}$')
plt.axvline(35.234, color='sandybrown', ls = '--', label = r'$\nu_-$')
plt.axvline(35.519, color='k', ls = '--', label = r'$\nu_0$')
plt.axvline(35.803, color='sienna', ls = '--', label = r'$\nu_+$')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.show()
plt.savefig('x7.pdf')




F = 0.866 #cm^2
R_3 = 998 #Ohm

#Dy2O3
U_o, U_m, R_o, R_m = np.genfromtxt('bdy.txt', unpack=True)
U_o/=10
U_m/=10

R_o = R_o*5/1000 #Ohm
R_m = R_m*5/1000 #Ohm

dR = R_o - R_m 
dU = U_m - U_o

Q = 15.1/(14.8*7.8) #cm^2


x_U = 4*F*dU/(Q*496) #???
x_R = 2*dR*F/(R_3*Q)

print('--------------Dy2O3-------------')
#print(f"R_o = {R_o}")
#print(f"R_m = {R_m}")
#print(f"dR = {dR}")
#print(f"dU = {dU}")
print(f"x_U = {np.mean(x_U)} pm {np.std(x_U)}")
print(f"x_R = {np.mean(x_R)} pm {np.std(x_R)}")

#Gd2O3
U_o, U_m, R_o, R_m = np.genfromtxt('bgd.txt', unpack=True)
U_o/=10
U_m/=10

R_o = R_o*5/1000 #Ohm
R_m = R_m*5/1000 #Ohm

dR = R_o - R_m 
dU = U_m - U_o



Q = 14.08/(15.5*7.4) #cm^2

x_U = 4*F*dU/(Q*496) #???
x_R = 2*dR*F/(R_3*Q)


print('--------------Gd2O3------------')
#print(f"R_o = {R_o}")
#print(f"R_m = {R_m}")
#print(f"dR = {dR}")
print(f"dU = {dU}")
print(f"x_U = {np.mean(x_U)} pm {np.std(x_U)}")
print(f"x_R = {np.mean(x_R)} pm {np.std(x_R)}")


L  = np.array([5, 0])
S  = np.array([5/2, 7/2])
J  = L + S
g_j = np.array([4/3, 2])
mu_b = 0.5*e*hbar/m
rho = np.array([7.8, 7.4]) * 1e3
M  = np.array([372.998, 336.49]) * 1e-3
N = 2*rho*N_A/M
T = 293.15 #K

x_T = mu_0*mu_b**2*g_j**2*N*J*(J+1)/(3*k*T)
print('--------------x_T------------')
print(x_T)
print('--------------N------------')
print(N)

print('--------------diskussion------------')

dx_U = (-0.017298983635975218+0.02540839)/0.02540839
dx_R = (0.02082149536689272-0.01485359)/0.01485359
print(f"dy2o3: dx_U = {dx_U} dx_R = {dx_R}")

dx_U = (-0.00785120028409091+0.02540839)/0.02540839
dx_R = (-0.010876587531122852+0.01485359)/0.01485359
print(f"gd2o3: dx_U = {dx_U} dx_R = {dx_R}")
dq = 1-q/100
print(f"dq: {dq}")