# ' ' = " "
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import sympy
from scipy.optimize import curve_fit
from uncertainties import ufloat


plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 16

t, T_1, p_1, T_2, p_2, N = np.genfromtxt('daten.txt', unpack=True)

p_1 += 1 #Auf die Drücke muss jeweils 1 bar addiert werden (siehe Aufgabenstellung)
p_2 += 1

t *= 60 #min in s

T_1 += 273.15 # Von Ceclsius in Kelvin
T_2 += 273.15

print(f"\n \n \n Aufgabe b):")

def f(x,A,B,C): #Funktion an die gefittet wird
    return A*x**2 + B*x + C

parameter1, covariance_matrix_1 = curve_fit(f,t,T_1) #Aufruf der Fitfunktion
parameter2, covariance_matrix_2 = curve_fit(f,t,T_2)

uncertainties1 = np.sqrt(np.diag(covariance_matrix_1))
uncertainties2 = np.sqrt(np.diag(covariance_matrix_2))

for names ,value, uncertaintie in zip("ABC", parameter1,uncertainties1):
    names = ufloat(value, uncertaintie)
    print(f"Die ersten drei Werte für T_1 mit Fehler sind: \n {names:.8f}")
for names ,value, uncertaintie in zip("DEF", parameter2,uncertainties2):
    names = ufloat(value, uncertaintie)
    print(f"Die ersten drei Werte  für T_2 mit Fehler sind: \n {names:.8f}")


# c) 4 Temperaturen t= 8, 16, 24, 32  

print(f"\n \n \n Aufgabe c):")

x = sympy.var('x')
T_1f = f(x, *parameter1)
T_2f = f(x, *parameter2)

T_1f_1 = T_1f.diff(x)
T_2f_1 = T_2f.diff(x)

cash = np.array([0., 0., 0., 0., 0., 0., 0., 0.]) #Array mit den Ergebnissen des einsetzens 0-3 T_1f_1 und 4-7 T_2f_1
#weil ich nicht herausbekkomen hab wie man zahlen einsetzt die quick and unrelayable version
#Für dT_1/dt gilt:

#Eleganter ist es, an den Kurvenverlauf ein Polynom 2ten Grades anzupassen und dieses Polynom
#zu differenzieren
for value in range(1,5):
    cash[value - 1] = parameter1[1] + 2* parameter1[0] * (8*value*60)
#    print(f"Das Ergebnis für T_1f_1(t={8*value}) = {cash[value - 1]} ")

#Für dT_2/dt gilt:
for value in range(1,5):
    cash[value+3] = parameter2[1] +2*parameter2[0] * (8*value*60)
#    print(f"Das Ergebnis für T_2f_1(t={8*value}) = {cash[value+3]} ")




#mit dem ansatz von bene und niklas
t_v, a_v, b_v , c_v=sympy.var("t_v a_v b_v c_v")
ae_v, be_v , ce_v=sympy.var("ae_v be_v ce_v")

T1test= a_v *t_v**2 + b_v *t_v + c_v

def F1dt(t):
    F1=T1test.diff(t_v)
    F1dt_value = F1.evalf(subs={t_v:t,a_v:parameter1[0], b_v:parameter1[1] , c_v:parameter1[2] })

    #mit fehlerfortpflanzung
    F1_error = F1.diff(a_v) * ae_v + F1.diff(b_v) * be_v + F1.diff(c_v) * ce_v
    F1dt_error = F1_error.evalf(subs={t_v:t, ae_v: uncertainties1[0], be_v: uncertainties1[1], ce_v: uncertainties1[2]})
    
    return ufloat(F1dt_value, F1dt_error)

def F2dt(t):
    F1=T1test.diff(t_v)
    F2dt_value = F1.evalf(subs={t_v:t,a_v:parameter2[0], b_v:parameter2[1] , c_v:parameter2[2] })

    #mit fehlerfortpflanzung
    F2_error = F1.diff(a_v) * ae_v + F1.diff(b_v) * be_v + F1.diff(c_v) * ce_v
    F2dt_error = F2_error.evalf(subs={t_v:t, ae_v: uncertainties1[0], be_v: uncertainties1[1], ce_v: uncertainties1[2]})
    return ufloat(F2dt_value, F2dt_error)




for i in range(1,5):
    print(f"F1dt({8*i})={F1dt(t[8*i-1]):.8f} , F2dt({8*i})={F2dt(t[8*i-1]):.8f}")




print(f"\n \n \n Aufgabe d): ")

m_kg_k = 750#J/K mc? # es fehlt die wärmekapazität des eimers
m = 4 #kg
c_w = 4183 #J/kg/K

def M_r(t):
    return ((m*c_w + m_kg_k)*cash[t])/(N[(t+1)*8])

def M_t(t):
    return (T_1[(t+1)*8])/(T_1[(t+1)*8] - T_2[(t+1)*8])

for i in range(0,4):
    print(f"Die reale Güteziffer bei t={(i+1)*8} ist: \n {((m*c_w + m_kg_k)*cash[i])/(N[(i+1)*8])} \nDie theoretische Güteziffer bei t={(i+1)*8} ist: \n {(T_1[(i+1)*8])/(T_1[(i+1)*8] - T_2[(i+1)*8])}")
   # print(N[(i+1)*8])


print(f"\n \n \n Aufgabe e): ")

L = 0#Verdampfungswärme




#Grafik
#plt.plot(t,T_1,'.', label=r"$Eimer 1$")
plt.xlabel(r'$t \, [s]$')
plt.ylabel(r'$T  \, [K]$')

plt.plot(t, f(t, *parameter1), label=r'$Fit \, T_1(t)$')
plt.errorbar(t, T_1, xerr=0, yerr=0.1, fmt='.', label = r'$T_1(t)$')

plt.plot(t, f(t, *parameter2), label=r'$Fit \, T_2(t)$')
plt.errorbar(t, T_2, xerr=0, yerr=.1, fmt='.', label = r'$T_2(t)$')

plt.legend()
plt.tight_layout()
plt.savefig('grafic.pdf')