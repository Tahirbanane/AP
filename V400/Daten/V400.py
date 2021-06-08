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
r  = np.array([81.5, 61, 50, 41.6, 36]) 
g  = np.array([83.5, 61.5, 50.5, 42.4, 36.7])   

print(f'------------------rot------------------')
for i in range(5):
    print(a1[i] + r[i] - 60)

print(f'------------------grün------------------')
for i in range(5):
    print(a1[i] + g[i] - 60)


print(f'\n ----------------------mit Brechungsindex----------------------')

nKron = 1.51673
b1 = np.zeros(5)
b2r = np.zeros(5)
b2g = np.zeros(5)

for i in range(5):
    b1[i] = np.arcsin(np.sin(np.deg2rad(a1[i])/nKron))

for i in range(5):
    b2r[i] = np.arcsin(np.sin(np.deg2rad(r[i])/nKron))

for i in range(5):
    b2g[i] = np.arcsin(np.sin(np.deg2rad(g[i])/nKron))

print(np.rad2deg(b1))
print(np.rad2deg(b2r))
print(np.rad2deg(b2g))