import matplotlib.pyplot as plt
import math
import matplotlib.colors as mcolors
import numpy as np
import uncertainties.unumpy as unp
import sympy
import statistics
from scipy.optimize import curve_fit
from uncertainties import ufloat
from scipy.signal import find_peaks
import scipy.constants as const


print (f'\n -----------------------------------Brechung-----------------------------------')

#Messwerte

x1 = ufloat(np.deg2rad(7),np.deg2rad(0.5))
x2 = ufloat(np.deg2rad(13.5),np.deg2rad(0.5))
x3 = ufloat(np.deg2rad(20),np.deg2rad(0.5))
x4 = ufloat(np.deg2rad(25),np.deg2rad(0.5))
x5 = ufloat(np.deg2rad(31),np.deg2rad(0.5))
x6 = ufloat(np.deg2rad(35.5),np.deg2rad(0.5))
x7 = ufloat(np.deg2rad(39),np.deg2rad(0.5))

b = np.array([x1, x2, x3, x4, x5, x6, x7])
print(b)

n1 = (np.sin(np.deg2rad(10))/unp.sin(x1))
n2 = (np.sin(np.deg2rad(20))/unp.sin(x2))
n3 = (np.sin(np.deg2rad(30))/unp.sin(x3))
n4 = (np.sin(np.deg2rad(40))/unp.sin(x4))
n5 = (np.sin(np.deg2rad(50))/unp.sin(x5))
n6 = (np.sin(np.deg2rad(60))/unp.sin(x6))
n7 = (np.sin(np.deg2rad(70))/unp.sin(x7))

print(n1,n2,n3,n4,n5,n6,n7)

n = (n1+n2+n3+n4+n5+n6+n7)/7

print(n)

c = 2.9979 * 1e8
print(c/n)


print (f'\n -----------------------------------Strahlversatz-----------------------------------')

d = 5.85 #cm
d *= 1e-2

s1 = d*(unp.sin(np.deg2rad(10)-x1)/unp.cos(x1))
s2 = d*(unp.sin(np.deg2rad(20)-x2)/unp.cos(x2))
s3 = d*(unp.sin(np.deg2rad(30)-x3)/unp.cos(x3))
s4 = d*(unp.sin(np.deg2rad(40)-x4)/unp.cos(x4))
s5 = d*(unp.sin(np.deg2rad(50)-x5)/unp.cos(x5))
s6 = d*(unp.sin(np.deg2rad(60)-x6)/unp.cos(x6))
s7 = d*(unp.sin(np.deg2rad(70)-x7)/unp.cos(x7))

print(s1,s2,s3,s4,s5,s6,s7)


print(f'\n -----------------------------------errechnetes Beta-----------------------------------')

b1 = unp.arcsin(np.sin(np.deg2rad(10))/n)
b2 = unp.arcsin(np.sin(np.deg2rad(20))/n)
b3 = unp.arcsin(np.sin(np.deg2rad(30))/n)
b4 = unp.arcsin(np.sin(np.deg2rad(40))/n)
b5=  unp.arcsin(np.sin(np.deg2rad(50))/n)
b6 = unp.arcsin(np.sin(np.deg2rad(60))/n)
b7 = unp.arcsin(np.sin(np.deg2rad(70))/n)



print(b1,b2,b3,b4,b5,b6,b7)
print(np.rad2deg(0.1178),'$ \pm $', np.rad2deg(0.0015) )
print(np.rad2deg(0.2336),'$ \pm $', np.rad2deg(0.0029) )
print(np.rad2deg(0.345), '$ \pm $', np.rad2deg(0.004) )
print(np.rad2deg(0.450), '$ \pm $', np.rad2deg(0.006) )
print(np.rad2deg(0.545), '$ \pm $', np.rad2deg(0.007) )
print(np.rad2deg(0.626), '$ \pm $', np.rad2deg(0.009) )
print(np.rad2deg(0.689), '$ \pm $', np.rad2deg(0.010) )

#print(np.rad2deg(b1),np.rad2deg(b2),np.rad2deg(b3),np.rad2deg(b4),np.rad2deg(b5),np.rad2deg(b6),np.rad2deg(b7))

s1 = d*(unp.sin(np.deg2rad(10)-b1)/unp.cos(b1))
s2 = d*(unp.sin(np.deg2rad(20)-b2)/unp.cos(b2))
s3 = d*(unp.sin(np.deg2rad(30)-b3)/unp.cos(b3))
s4 = d*(unp.sin(np.deg2rad(40)-b4)/unp.cos(b4))
s5 = d*(unp.sin(np.deg2rad(50)-b5)/unp.cos(b5))
s6 = d*(unp.sin(np.deg2rad(60)-b6)/unp.cos(b6))
s7 = d*(unp.sin(np.deg2rad(70)-b7)/unp.cos(b7))

print(s1,s2,s3,s4,s5,s6,s7)

print(f'\n -----------------------------------Prisma-----------------------------------')



#Messwerte #Fehlerergänzen
a1 = np.array([30,40,50,60,70])
r  = np.array([81.5,61, 50,41.6, 36]) 
g  = np.array([83.5,61.5, 50.5,42.4,36.7])   

print(f'------------------rot------------------')
for i in range(5):
    print(a1[i] + r[i] - 60)

print(f'------------------grün------------------')
for i in range(5):
    print(a1[i] + g[i] - 60)


print(f'\n ----------------------mit Brechungsindex----------------------') #np.array([ufloat(0,0.5),ufloat(0,0.5),ufloat(0,0.5),ufloat(0,0.5),ufloat(0,0.5) ])

nKron = 1.51673
n = nKron
#zero = np.zeros(5)
#b1 =  np.array(zero)
#b2r = np.array(zero)
#b2g = np.array(zero)
#
#for i in range(5):
#    b1[i] = np.arcsin(np.sin(np.deg2rad(a1[i])/nKron))
#
#for i in range(5):
#    b2r[i] = np.arcsin(np.sin(np.deg2rad(r[i])/nKron))
#
#for i in range(5):
#    b2g[i] = np.arcsin(np.sin(np.deg2rad(g[i])/nKron))
#
#b1 = (np.rad2deg(b1))
#b2r = (np.rad2deg(b2r))
#b2g = (np.rad2deg(b2g))
#
#dr = np.array(zero)
#dg = np.array(zero)
#
#for i in range(5):
#    dr[i] = a1[i] + r[i] - b1[i] + b2r[i]
#
#for i in range(5):
#    dg[i] = a1[i] + g[i] - b1[i] + b2g[i]
#
#print(dr)
#print(dg)

alpha1 = a1
alpha1 = np.array(alpha1)
alpha1 = alpha1 * 2 * np.pi / 360

alpha2red = r
alpha2red = np.array(alpha2red)
alpha2red = alpha2red * 2 * np.pi / 360

alpha2green = g
alpha2green = np.array(alpha2green)
alpha2green = alpha2green * 2 * np.pi / 360

# Berechnung der beta
beta1 = np.arcsin(np.sin(alpha1) / n)
beta2red = np.arcsin(np.sin(alpha2red) / n)
beta2green = np.arcsin(np.sin(alpha2green) / n)

# berechnung von delta
delta_red = (alpha1 + alpha2red) - beta1 - beta2red
delta_green = (alpha1 + alpha2green) - beta1 - beta2green

# Rückkonvertierung nach Grad
conv = 360 / (2 * np.pi) # conversion rad -> deg
alpha1 = alpha1 * conv
alpha2red = alpha2red * conv
alpha2green = alpha2green * conv
beta1 = beta1 * conv
beta2red = beta2red * conv
beta2green = beta2green * conv
delta_red = delta_red * conv
delta_green = delta_green * conv

print(delta_red)
print(delta_green)

print(f'\n -----------------------------------Gitter-----------------------------------')

conv = 2 * np.pi / 360 # conversion deg->rad

# 100 Linien/mm
d = 10 * 10**(-6) # gitterkonstante in SI

gruen1 = [3.1, 6.3, 9.2, 12.5, 15.9]  
gruen2 = [3.1, 6,  9.4, 12.4, 16.5]
kgruen = np.arange(1, 6)
gruen = np.append(gruen1, gruen2)
kgruen = np.append(kgruen, kgruen)

rot1 = [3.5, 7.2, 11, 14.8, 18.7]  
rot2 = [3.9, 7.5, 11.2, 15.1, 18.9]
krot = np.arange(1, 6)
rot = np.append(rot1, rot2)
krot = np.append(krot, krot)

# conv nach radiants
rot = rot * conv
gruen = gruen * conv

lambda_rot = d * np.divide(np.sin(rot), krot)
lambda_gruen = d * np.divide(np.sin(gruen), kgruen)

lambda_rot_std = np.std(lambda_rot)
lambda_gruen_std = np.std(lambda_gruen)
lambda_rot = np.mean(lambda_rot)
lambda_gruen = np.mean(lambda_gruen)

print(f'100 Linien/mm')
print(f'lambda_rot = {lambda_rot} \pm {lambda_rot_std}')
print(f'lambda_gruen = {lambda_gruen} \pm {lambda_gruen_std}')

lambda_rot_100 = ufloat(lambda_rot, lambda_rot_std)
lambda_gruen_100 = ufloat(lambda_gruen, lambda_gruen_std)

# 300 Linien/mm
d = 3.3 * 10**(-6) # gitterkonstante in SI

gruen1 = [9.3, 18.8, 28.5]
gruen2 = [9, 18.5  , 28.5]
kgruen = np.arange(1, 4)
gruen = np.append(gruen1, gruen2)
kgruen = np.append(kgruen, kgruen)

rot1 = [10.8, 22 ,34.2]
rot2 = [10.9, 22.25, 35]
krot = np.arange(1, 4)
rot = np.append(rot1, rot2)
krot = np.append(krot, krot)

rot = rot * conv
gruen = gruen * conv

lambda_rot = d * np.divide(np.sin(rot), krot)
lambda_gruen = d * np.divide(np.sin(gruen), kgruen)

lambda_rot_std = np.std(lambda_rot)
lambda_gruen_std = np.std(lambda_gruen)
lambda_rot = np.mean(lambda_rot)
lambda_gruen = np.mean(lambda_gruen)

print(f'300 Linien/mm')
print(f'lambda_rot = {lambda_rot} \pm {lambda_rot_std}')
print(f'lambda_gruen = {lambda_gruen} \pm {lambda_gruen_std}')

lambda_rot_300 = ufloat(lambda_rot, lambda_rot_std)
lambda_gruen_300 = ufloat(lambda_gruen, lambda_gruen_std)

# 600 Linien/mm
d = 1.67 * 10**(-6) # gitterkonstante in SI

gruen1 = [19]
gruen2 = [19]
kgruen = np.arange(1, 2)
gruen = np.append(gruen1, gruen2)
kgruen = np.append(kgruen, kgruen)

rot1 = [22.5]
rot2 = [23]
krot = np.arange(1, 2)
rot = np.append(rot1, rot2)
krot = np.append(krot, krot)

rot = rot * conv
gruen = gruen * conv

lambda_rot = d * np.divide(np.sin(rot), krot)
lambda_gruen = d * np.divide(np.sin(gruen), kgruen)


lambda_rot_std = np.std(lambda_rot)
lambda_gruen_std = np.std(lambda_gruen)
lambda_rot = np.mean(lambda_rot)
lambda_gruen = np.mean(lambda_gruen)

print(f'600 Linien/mm')
print(f'lambda_rot = {lambda_rot} \pm {lambda_rot_std}')
print(f'lambda_gruen = {lambda_gruen} \pm {lambda_gruen_std}')

lambda_rot_600 = ufloat(lambda_rot, lambda_rot_std)
lambda_gruen_600 = ufloat(lambda_gruen, lambda_gruen_std)

# calculate overall average
lambda_rot_overall = (lambda_rot_100 + lambda_rot_300 + lambda_rot_600) / 3
lambda_gruen_overall = (lambda_gruen_100 + lambda_gruen_300 + lambda_gruen_600) / 3

print(f'Overall average')
print(f'lambda_rot = {lambda_rot_overall}')
print(f'lambda_gruen = {lambda_gruen_overall}')
