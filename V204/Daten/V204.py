import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import sympy
import statistics
from scipy.optimize import curve_fit
from uncertainties import ufloat
from scipy.signal import find_peaks


t_1, T_1_1, T_4_1, T_5_1, T_8_1, T_2_1, T_3_1, T_6_1, T_7_1 = np.genfromtxt('Daten/statisch.txt', unpack=True) #Daten/
t_2, T_1_2, T_2_2, T_3_2, T_4_2, T_5_2, T_6_2, T_7_2, T_8_2 = np.genfromtxt('Daten/dynamisch_80s.txt', unpack=True)
t_3, T_1_3, T_2_3, T_3_3, T_4_3, T_5_3, T_6_3, T_7_3, T_8_3 = np.genfromtxt('Daten/dynamisch_200s.txt', unpack=True)

t_1 /= 10 #auf sekunden
t_2 *= 2
t_3 *= 2

T_1_1 += 273.15
T_4_1 += 273.15
T_5_1 += 273.15
T_8_1 += 273.15
T_2_1 += 273.15
T_3_1 += 273.15
T_6_1 += 273.15
T_7_1 += 273.15

T_1_2 += 273.15
T_4_2 += 273.15
T_5_2 += 273.15
T_8_2 += 273.15
T_2_2 += 273.15
T_3_2 += 273.15
T_6_2 += 273.15
T_7_2 += 273.15

T_1_3 += 273.15
T_4_3 += 273.15
T_5_3 += 273.15
T_8_3 += 273.15
T_2_3 += 273.15
T_3_3 += 273.15
T_6_3 += 273.15
T_7_3 += 273.15


#statische Methode Plot:
plt.figure()
plt.xlabel(r'$t \, [s]$')
plt.ylabel(r'$T  \, [K]$')

plt.plot(t_1, T_1_1, label=r'$T1 = Messing(dick)$')
plt.plot(t_1, T_4_1, label=r'$T4 = Messing(dünn)$')
plt.plot(t_1, T_5_1, label=r'$T5 = Aluminium$')
plt.plot(t_1, T_8_1, label=r'$T8 = Edelstahl$')
plt.legend()
plt.tight_layout()
plt.savefig('Daten/grafic.pdf')

#statische Methode 700s:
print(f'Die Temperatur bei T1 beträgt: {T_1_1[7000]}, \n Die Temperatur bei T4 beträgt: {T_4_1[7000]}, \n Die Temperatur bei T5 beträgt: {T_5_1[7000]}, \n Die Temperatur bei T8 beträgt: {T_1_1[7000]}, \n')

#Wärmestrom:
def kappa(roh,c, x, t, A_nah, A_fern):
    return (roh*c*(x**2))/(2*t*np.log(A_nah/A_fern)) #hier ist A die Amplitude

#Amplitudenberechnung:
def A(T):
    x, _ = find_peaks(T)
    T *= -1
    y, _ = find_peaks(T)
    T *= -1
    if len(x) < len(y):
        l += T[x] - T[y[0:len(x)]]
        for i in range(0,len(x)-1):
            k += l[i]
        return l/len(x)

    if len(x) > len(y):
        l += T[y] - T[x[0:len(y)]]
        for i in range(0,len(y)-1):
            k += l[i]
        return l/len(y)

    else:
        l += T[x] - T[y]
        for i in range(0,len(x)-1):
            k += l[i]
        return l/len(x)

#Durchschnitt:
def ds(A):
    return statistics.mean(A)

#Betrag:
def betrag(A):
    for i in range(0,len(A)-1):
        if A[i] < 0:
            A[i] *= -1
    return A

def betrag_z(A):
    if(A < 0):
        return -A
    return A


def lam(kap, f, roh, c):
    np.sqrt((4*kap*np.pi*f)/(roh*c))


T_12_1 = T_1_1 - T_2_1

for i in range(0,5):
    print(f'Der Wärmestrom zum Zeitpunkt t={i*100} beträgt {-142*0.012*0.004*T_12_1[i*1000]/0.029} (Messing dick)')

# alle metalle machen T3 ist messing dünn
T_34_1 = T_3_1 - T_4_1

for i in range(0,5):
    print(f'Der Wärmestrom zum Zeitpunkt t={i*100} beträgt {-142*0.007*0.004*T_34_1[i*1000]/0.029} (Messing dünn)')

#Aluminium T5
T_56_1 = T_5_1 - T_6_1

for i in range(0,5):
    print(f'Der Wärmestrom zum Zeitpunkt t={i*100} beträgt {-221*0.012*0.004*T_56_1[i*1000]/0.03} (Aluminium)')



# Edelstahl T7
T_78_1 = T_7_1 - T_8_1

# da es um das Verhältnis der Amplituden geht ist es egal ob es sich hierbei um die doppelte Amplitude handelt oder nicht
for i in range(0,5):
    print(f'Der Wärmestrom zum Zeitpunkt t={i*100} beträgt {-21*0.012*0.004*T_78_1[i*1000]/0.031} (Edelstahl)') 

#print(f'{T_1_1[0]} \n {T_1_1[-1]} \n {T_1_1[7771]} \n {T_1_1[-1]} \n {T_1_1[0:3-4]} \n {x} ')


# T7- T8 und T2 - T1

plt.figure()
plt.xlabel(r'$t \, [s]$')
plt.ylabel(r'$T  \, [K]$')

plt.plot(t_1, T_7_1- T_8_1, label=r'$T7- T8$')

plt.legend()
plt.tight_layout()
plt.savefig('grafic1.pdf')


plt.figure()
plt.xlabel(r'$t \, [s]$')
plt.ylabel(r'$T  \, [K]$')

plt.plot(t_1, T_2_1 - T_1_1, label=r'$T2 - T1$')

plt.legend()
plt.tight_layout()
plt.savefig('grafic2.pdf')



# Dynamisch Periode 80s

plt.figure()
plt.xlabel(r'$t \, [s]$')
plt.ylabel(r'$T  \, [K]$')

plt.plot(t_2, T_1_2, label=r'$T1$')
plt.plot(t_2, T_2_2, label=r'$T2$')

plt.legend()
plt.tight_layout()
plt.savefig('grafic3.pdf')

plt.figure()
plt.xlabel(r'$t \, [s]$')
plt.ylabel(r'$T  \, [K]$')

plt.plot(t_2, T_5_2, label=r'$T5$')
plt.plot(t_2, T_6_2, label=r'$T6$')

plt.legend()
plt.tight_layout()
plt.savefig('grafic5.pdf')


#Messing
#Berechnung von der Amplitude T1
x1, _ = find_peaks(T_1_2, distance = 15)
x2, _ = find_peaks(-T_1_2, distance = 15)
A11 = T_1_2[x1] - T_1_2[x2]
A1 = ds(A11)
print(f'Die mittlere Amplitude von T1 beträgt A1 = {A11} ')

#Berechnung von der Amplitude T2
x3, _ = find_peaks(T_2_2, distance = 15)
x4, _ = find_peaks(-T_2_2, distance = 15)
A22 = T_2_2[x3] - T_2_2[x4]
A2 = ds(A22)
print(f'Die Amplitude von T2 beträgt A2 = {A22} ')

#Berechnung der mittleren Phasendifferenz

t_ph = (t_2[x1] - t_2[x3] + t_2[x2] - t_2[x4])/2


print(f'Die Phasendifferenz beträgt t = {t_ph}')

#Berechnung der mittleren Wärmeleitfähigkeit

print(f'Die Wärmeleitfähigkeit beträgt k = {kappa(8400, 377, 0.029, t_ph, A11, A22) } (Messing dick) ' )

#Aluminium
#Berechnung von der Amplitude T5
x1, _ = find_peaks(T_5_2, distance = 15)
x2, _ = find_peaks(-T_5_2, distance = 15)
A55 = T_5_2[x1][0:-2] - T_5_2[x2][0:-2]
A5 = ds(A55)
print(f'Die mittlere Amplitude von T5 beträgt A5 = {A5} ')

#Berechnung von der Amplitude T6
x3, _ = find_peaks(T_6_2, distance = 15)
x4, _ = find_peaks(-T_6_2, distance = 15)
A66 = T_6_2[x3][0:-3] - T_6_2[x4][0:-2]
A6 = ds(A66)
print(f'Die mittlere Amplitude von T6 beträgt A6 = {A6} ')

#Berechnung der mittleren Phasendifferenz
t_ph1 = (t_2[x1] - t_2[x3])
t_ph2 = (t_2[x2][0:-1] - t_2[x4])
t_ph = (t_ph1[0:-1] + t_ph2)/2


print(f'Die durschnittliche Phasendifferenz beträgt t = {ds(t_ph)}')

#Berechnung der mittleren Wärmeleitfähigkeit

print(f'Die mittleren Wärmeleitfähigkeit beträgt k = {kappa(2700, 896, 0.03, ds(t_ph), A5, A6) } (Aluminium) ' )
print(f'Die mittleren Wellenlänge beträgt lambda = {lam(kappa(2700, 896, 0.03, ds(t_ph), A5, A6),0.01228,2700,896) } (Aluminium)')


# Dynamisch Periode 200s

plt.figure()
plt.xlabel(r'$t \, [s]$')
plt.ylabel(r'$T  \, [K]$')

plt.plot(t_3, T_7_3, label=r'$T7$')
plt.plot(t_3, T_8_3, label=r'$T8$')

plt.legend()
plt.tight_layout()
plt.savefig('grafic4.pdf')


#Amplituden

#Berechnung von der Amplitude T7
x1, _ = find_peaks(T_7_3, distance = 15)
x2, _ = find_peaks(-T_7_3, distance = 15)
A77 = T_7_3[x1][0:-1] - T_7_3[x2]
A7 = ds(A77)

print(f'Die mittlere Amplitude von T7 beträgt A7 = {A77} ')



#Berechnung von der Amplitude T6
x3, _ = find_peaks(T_8_3, distance = 15)
x4, _ = find_peaks(-T_8_3, distance = 15)

#T_8_3[x3] = [301.93, 308.58, 314.37, 319.04, 322.85]
#T_8_3[x4] = [294.36, 301.86, 308.57, 313.84, 318.18]

x3 = [94, 190, 287, 384]
x4 = [109, 213, 316, 418]

A88 = T_8_3[x3] - T_8_3[x4]
A8 = ds(A88)


print(f'Die mittlere Amplitude von T8 beträgt A8 = {A88} ')

#Berechnung der mittleren Phasendifferenz

t_ph1 = (t_3[x1][0:-1] - t_3[x3])
t_ph2 = (t_3[x2] - t_3[x4])
t_ph = (t_ph1 + t_ph2)/2

print(f'Die Phasendifferenz beträgt t = {t_ph}')
print(f'Die durschnittliche Phasendifferenz beträgt t = {ds(t_ph)}')

#Berechnung der Wärmeleitfähigkeit

print(f'Die Anzahl an ermittelten Perioden ist {len(A77)} Wärmeleitfähigkeit beträgt k = {kappa(8000, 500, 0.031, ds(t_ph), A77, A88) } (Edelstahl) ' )