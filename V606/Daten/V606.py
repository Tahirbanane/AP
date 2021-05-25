import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import scipy.constants as const

e=const.e
h=const.hbar
m=const.m_e
mu_0=const.mu_0
k=const.k
N_A=const.N_A

f, U_A = np.genfromtxt('a.txt', unpack=True)
plt.plot(f, U_A/496,'k.', label ='Messwerte')
plt.xlabel(r'Frequenz $\nu$ / $\mathrm{kHz}$')
plt.ylabel(r'Spannungsverhältnis $\frac{U_A}{U_E}$')
plt.legend()
plt.tight_layout()
plt.savefig('selektiv.pdf')


F = 0.866 #cm^2
R_3 = 998 #Ohm

#Dy2O3
U_o, U_m, R_o, R_m = np.genfromtxt('bdy.txt', unpack=True)

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
#print(f"dU = {dU}")
print(f"x_U = {np.mean(x_U)} pm {np.std(x_U)}")
print(f"x_R = {np.mean(x_R)} pm {np.std(x_R)}")


L  = np.array([5, 0])
S  = np.array([5/2, 7/2])
J  = L + S
g_j = np.array([4/3, 2])
mu_b = 0.5*e*h/m
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

dx_U = (0.17298983635975218-0.02540839)/0.02540839
dx_R = (0.02082149536689272-0.01485359)/0.01485359
print(f"dy2o3: dx_U = {dx_U} dx_R = {dx_R}")

dx_U = (0.0785120028409091-0.02540839)/0.02540839
dx_R = (-0.010876587531122852+0.01485359)/0.01485359
print(f"gd2o3: dx_U = {dx_U} dx_R = {dx_R}")