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
from scipy.interpolate import UnivariateSpline

e=const.e
h=const.h
c=const.c
d_LiF=201.4*10**(-12) #m
ry=13.6 #eV
alpha=const.value("fine-structure constant")

print (f'\n -----------------------------------Bragg-Bedingung-----------------------------------')

theta_bragg,N_bragg=np.genfromtxt('bragg.txt', unpack = True)
plt.plot(theta_bragg, N_bragg,'kx',label = 'Messdaten')
plt.ylabel(r'$N \,/\, \mathrm{\frac{Imp}{s}}$')
N_max = find_peaks(N_bragg, height=100)
N_peak_bragg = N_bragg[N_max[0]]
theta_peak_bragg = theta_bragg[N_max[0]]
plt.plot(theta_peak_bragg,N_peak_bragg,'r.',label = 'Maximum')
plt.xlabel(r'$\theta \,/\, °$')
plt.legend()
plt.tight_layout()
plt.savefig("bragg.pdf")
plt.close()

print(f"maximale impulsrate von {N_peak_bragg} bei {theta_peak_bragg}°")

print (f'\n -----------------------------------Emissionsspektrum-----------------------------------')

theta_cu,N_cu = np.genfromtxt('emissionsspektrum.txt', unpack = True)
N_max = find_peaks(N_cu, height=1000)
N_peak_cu = N_cu[N_max[0]]
theta_peak_cu = theta_cu[N_max[0]]
plt.plot(theta_cu, N_cu,'kx',label = 'Messdaten')
plt.ylabel(r'$N \,/\, \mathrm{\frac{Imp}{s}}$')
plt.plot(theta_cu[0:120],N_cu[0:120],'r--',label = 'Bremsberg')
plt.plot(theta_cu[127:143],N_cu[127:143],'r--')
plt.plot(theta_cu[150:180],N_cu[150:180],'r--')
plt.axvline(theta_peak_cu[0], color='steelblue', ls = '--', label = r'$K_{\mathrm{\beta}}$')
plt.axvline(theta_peak_cu[1], color='g', ls = '--', label = r'$K_{\mathrm{\alpha}}$')
plt.xlabel(r'$\theta \,/\, °$')
plt.legend()
plt.tight_layout()
plt.savefig("emissionsspektrum.pdf")
plt.close()
#bremsberg einfügen


#def f(x, c, d, e, f): 
#    return c*np.exp(x*d) + e*np.exp(x*f)
#
#
#def G(x, A, x0, sigma1, B, x1, sigma2):
#    return A * np.exp(-(x-x0)**2/(2*sigma1**2)) +  B * np.exp(-(x-x1)**2/(2*sigma2**2))
#
#sigma_1 = np.std(theta_cu[120:126], ddof = 1) * 1 / len(theta_cu[120:126])
#sigma_2 = np.std(theta_cu[143:150], ddof = 1) * 1 / len(theta_cu[143:150])
#
#params,covariance_matrix = curve_fit(G, theta_cu, N_cu, p0 =[4000, 20, theta_peak_cu[0], 160000, 22, theta_peak_cu[1]])
#print(f"{params}")
#
#
#
#plt.figure()
#plt.plot(theta_cu, G(N_cu, *params))
#plt.plot(theta_cu, N_cu, 'r.')
#plt.savefig("test3.pdf")

print(f"k-linien bei {theta_peak_cu}°")

lambda_min=(h*c)/(e*35000)
E_max=35000
theta_grenz=np.arcsin(lambda_min/(2*d_LiF))*180/np.pi

print(f"Grenzwinkel {theta_grenz} Wellenlänge min {lambda_min} Energie max {E_max}")


theta_cub,N_cub=np.genfromtxt('emissionsspektrumb.txt', unpack = True)
theta_cubb,N_cubb=np.genfromtxt('emissionsspektrumbb.txt', unpack = True)
theta_cubbb,N_cubbb=np.genfromtxt('emissionsspektrumbbb.txt', unpack = True)
plt.plot(theta_cub,N_cub,'kx',label = 'Messdaten')
plt.axvline(20.2, color='steelblue', ls = '--', label = r'$K_{\mathrm{\beta}}$')
plt.axvline(22.5, color='g', ls = '--', label = r'$K_{\mathrm{\alpha}}$')
spline = UnivariateSpline(theta_cubb, N_cubb-np.max(N_cubb)/2, s=0)
r1a, r2a = spline.roots()
plt.axvspan(r1a, r2a, facecolor='lightgreen', alpha=0.5)
spline = UnivariateSpline(theta_cubbb, N_cubbb-np.max(N_cubbb)/2, s=0)
r1b, r2b = spline.roots() 
plt.axvspan(r1b, r2b, facecolor='lightgreen', alpha=0.5)
plt.ylabel(r'$N \,/\, \mathrm{\frac{Imp}{s}}$')
plt.xlabel(r'$\theta \,/\, °$')
plt.legend()
plt.tight_layout()
plt.savefig("emissionsspektrum2.pdf")
plt.close()

FWHMa = (h*c)/(2*d_LiF*np.sin(r1a*np.pi/180))-(h*c)/(2*d_LiF*np.sin(r2a*np.pi/180))
FWHMa=FWHMa/const.e
FWHMb = (h*c)/(2*d_LiF*np.sin(r1b*np.pi/180))-(h*c)/(2*d_LiF*np.sin(r2b*np.pi/180))
FWHMb=FWHMb/const.e
E_K_peaka = (h*c)/(2*d_LiF*np.sin(theta_peak_cu[1]*np.pi/180))
E_K_peaka=E_K_peaka/const.e #eV
E_K_peakb = (h*c)/(2*d_LiF*np.sin(theta_peak_cu[0]*np.pi/180))
E_K_peakb=E_K_peakb/const.e #eV
Aa = E_K_peaka/FWHMa
Ab = E_K_peakb/FWHMb

E_K_a_lit = 8046 #ev
E_K_b_lit = 8904 #ev

E_K_a_rel = abs(E_K_a_lit-E_K_peaka)/E_K_peaka
E_K_b_rel = abs(E_K_b_lit-E_K_peakb)/E_K_peakb
theta_b_rel = abs(22.49-22.5)/22.49
theta_a_rel = (20.22-20.2)/20.22

z=29
E_K_lit =  8980.476 #eV
sigma_1 = z - np.sqrt(E_K_lit/ry) #°
sigma_2 = z - np.sqrt(4*(z-sigma_1)**2 - (4/ry)*E_K_peaka) #°
sigma_3 = z - np.sqrt(9*(z-sigma_1)**2 - (9/ry)*E_K_peakb) #°

print(f"sigma_1 = {sigma_1} , sigma_2 = {sigma_2} , sigma_3 = {sigma_3}")
print(f"E_K_alpha = {E_K_peaka}eV, E_K_beta = {E_K_peakb}eV ")
print(f"E_FWHM_alpha = {FWHMa}eV, E_FWHM_beta = {FWHMb}eV")
print(f"A_alpha = {Aa} A_beta = {Ab}")
print(f"rel. abw.: E_K_a = {E_K_a_rel}, E_K_b = {E_K_b_rel},theta_a_rel = {theta_a_rel}, theta_b_rel = {theta_b_rel}")

print (f'\n -----------------------------------Absorptionsspektrum-----------------------------------')

#zink
theta_zn,N_zn=np.genfromtxt('zink.txt', unpack = True)
plt.plot(theta_zn, N_zn,'kx',label = 'Messdaten')
plt.axvline(18.26666, color='steelblue', ls = '--', label = r'$N_{min}$')
plt.axvline(19.0, color='g', ls = '--', label = r'$N_{max}$')
plt.ylabel(r'$N \,/\, \mathrm{\frac{Imp}{s}}$')
plt.xlabel(r'$\theta \,/\, °$')
plt.legend()
plt.tight_layout()
plt.savefig("zink.pdf")
plt.close()
theta_zn=18.6
N_K_zn = N_zn[1]+(N_zn[10]-N_zn[1])/2
E_K_zn=(h*c)/(e*2*d_LiF*np.sin(18.6*np.pi/180))
E_K_zn_abs=9650-E_K_zn
E_K_zn_rel=E_K_zn_abs/9.65
sigma_K_zn=30-np.sqrt((E_K_zn/ry)-(alpha**2 * 30**4 /4))
sigma_K_zn_abs=sigma_K_zn-3.56
sigma_K_zn_rel=sigma_K_zn_abs/3.56
theta_zn_abs=18.6-18.60042993
theta_zn_rel=theta_zn_abs/18.60042993
print(f"E_K_zn = {E_K_zn} theta_zn = {theta_zn} sigma_K_zn = {sigma_K_zn} sigma abs {sigma_K_zn_abs} sigma rel {sigma_K_zn_rel} theta abs {theta_zn_abs} theta rel {theta_zn_rel} E abs {E_K_zn_abs} E rel = {E_K_zn_rel}")

#brom
theta_br,N_br=np.genfromtxt('brom.txt', unpack = True)
plt.plot(theta_br, N_br,'kx',label = 'Messdaten')
plt.axvline(13.0, color='steelblue', ls = '--', label = r'$N_{min}$')
plt.axvline(13.55, color='g', ls = '--', label = r'$N_{max}$')
plt.ylabel(r'$N \,/\, \mathrm{\frac{Imp}{s}}$')
plt.xlabel(r'$\theta \,/\, °$')
plt.legend()
plt.tight_layout()
plt.savefig("brom.pdf")
plt.close()
theta_br=13.3
N_K_br = N_br[2]+(N_br[7]-N_br[2])/2
E_K_br=(h*c)/(e*2*d_LiF*np.sin(13.2*np.pi/180))
sigma_K_br=35-np.sqrt((E_K_br/ry)-(alpha**2 * 35**4 /4))
print(f"E_K_br = {E_K_br} theta_br = {theta_br} sigma_K_br = {sigma_K_br}")


#gallium
theta_ga,N_ga=np.genfromtxt('gallium.txt', unpack = True)
plt.plot(theta_ga, N_ga,'kx',label = 'Messdaten')
plt.axvline(17.05, color='steelblue', ls = '--', label = r'$N_{min}$')
plt.axvline(17.85, color='g', ls = '--', label = r'$N_{max}$')
plt.ylabel(r'$N \,/\, \mathrm{\frac{Imp}{s}}$')
plt.xlabel(r'$\theta \,/\, °$')
plt.legend()
plt.tight_layout()
plt.savefig("gallium.pdf")
plt.close()
theta_ga=17.5
N_K_ga = N_ga[1]+(N_ga[8]-N_ga[1])/2
E_K_ga=(h*c)/(e*2*d_LiF*np.sin(17.3*np.pi/180))
sigma_K_ga=31-np.sqrt((E_K_ga/ry)-(alpha**2 * 31**4 /4))
print(f"E_K_ga = {E_K_ga} theta_ga = {theta_ga} sigma_K_ga = {sigma_K_ga}")


#rubidium
theta_rb,N_rb=np.genfromtxt('rubidium.txt', unpack = True)
plt.plot(theta_rb, N_rb,'kx',label = 'Messdaten')
plt.axvline(11.35, color='steelblue', ls = '--', label = r'$N_{min}$')
plt.axvline(12.1, color='g', ls = '--', label = r'$N_{max}$')
plt.ylabel(r'$N \,/\, \mathrm{\frac{Imp}{s}}$')
plt.xlabel(r'$\theta \,/\, °$')
plt.legend()
plt.tight_layout()
plt.savefig("rubidium.pdf")
plt.close()
theta_rb=11.7
N_K_rb = N_rb[3]+(N_rb[9]-N_rb[3])/2
E_K_rb=(h*c)/(e*2*d_LiF*np.sin(11.7*np.pi/180))
sigma_K_rb=37-np.sqrt((E_K_rb/ry)-(alpha**2 * 37**4 /4))
print(f"E_K_rb = {E_K_rb} theta_rb = {theta_rb} sigma_K_rb ={sigma_K_rb}")



#strontium
theta_sr,N_sr=np.genfromtxt('strontium.txt', unpack = True)
plt.plot(theta_sr, N_sr,'kx',label = 'Messdaten')
plt.axvline(10.7, color='steelblue', ls = '--', label = r'$N_{min}$')
plt.axvline(11.6, color='g', ls = '--', label = r'$N_{max}$')
plt.ylabel(r'$N \,/\, \mathrm{\frac{Imp}{s}}$')
plt.xlabel(r'$\theta \,/\, °$')
plt.legend()
plt.tight_layout()
plt.savefig("strontium.pdf")
plt.close()
theta_sr=11.2
N_K_sr = N_sr[2]+(N_sr[11]-N_sr[2])/2
E_K_sr=(h*c)/(e*2*d_LiF*np.sin(11.2*np.pi/180))
sigma_K_sr=38-np.sqrt((E_K_sr/ry)-(alpha**2 * 38**4 /4))
print(f"E_K_sr = {E_K_sr} theta_sr = {theta_sr} sigma_K_sr = {sigma_K_sr}")

#zirkonium
theta_zr,N_zr=np.genfromtxt('zirkonium.txt', unpack = True)
plt.plot(theta_zr, N_zr,'kx',label = 'Messdaten')
plt.axvline(9.5, color='steelblue', ls = '--', label = r'$N_{min}$')
plt.axvline(10.4, color='g', ls = '--', label = r'$N_{max}$')
plt.ylabel(r'$N \,/\, \mathrm{\frac{Imp}{s}}$')
plt.xlabel(r'$\theta \,/\, °$')
plt.legend()
plt.tight_layout()
plt.savefig("zirkonium.pdf")
plt.close()
theta_zr=10
N_K_zr = N_zr[0]+(N_zr[9]-N_zr[0])/2
E_K_zr=(h*c)/(e*2*d_LiF*np.sin(10*np.pi/180))
sigma_K_zr=40-np.sqrt((E_K_zr/ry)-(alpha**2 * 40**4 /4))
print(f"E_K_zr = {E_K_zr} theta_zr = {theta_zr} sigma_K_zr = {sigma_K_zr}")


print (f'\n -----------------------------------Moseleysches-Gesetz-----------------------------------')

Z = np.array ([30,31,35,37,38,40])
E_K = np.array ([E_K_zn,E_K_ga,E_K_br,E_K_rb,E_K_sr, E_K_zr])

params, covariance_matrix = np.polyfit(Z, np.sqrt(E_K), deg=1, cov=True)

x_plot = np.linspace(30, 40)
plt.plot(
    x_plot,
    params[0] * x_plot + params[1],
    color='steelblue',ls='-', label='Lineare Regression',
    linewidth=3,
   
)
plt.plot(Z, np.sqrt(E_K), 'kx',label="Messdaten")
plt.legend(loc="best")
plt.ylabel(f"Wurzel aus $E_K$")
plt.xlabel(f"Ordnungszahl Z")
plt.legend()
plt.tight_layout()
plt.savefig("mose.pdf")
plt.close()
rconst=(params[0]**2)*e/(h*c)

print(f"Ausgleichsgerade g={params[0]}x {params[1]}")
ry_m=(params[0]**2)/h
rconstabs=10973731.56816-((params[0]**2)*e/(h*c))
rconstrel=(10973731.56816-(params[0]**2)*e/(h*c))/10973731.56816


print(f"berechnete rydberg energie = {params[0]**2} rydberg const {rconst} ryconst abs{rconstabs} ryconst rel {rconstrel}")

print(f"---------------------------------Diskussion------------------------------")

E_K_lit= np.array([9650, 10370, 13470, 15200, 16100, 17990])
E_K_rel = abs(E_K-E_K_lit)/E_K_lit
print(f"relative abweichungen E_K: {E_K_rel}")

sigma_K = np.array ([sigma_K_zn,sigma_K_ga,sigma_K_br,sigma_K_rb,sigma_K_sr, sigma_K_zr])
sigma_K_lit = np.array ([3.56, 3.68, 3.85, 3.95, 4.01, 4.11])
sigma_K_rel = abs(sigma_K-sigma_K_lit)/sigma_K_lit
print(f"relative abweichungen sigma_K: {sigma_K_rel}")

theta = np.array ([theta_zn,theta_ga,theta_br,theta_rb,theta_sr, theta_zr])
theta_lit = np.array ([ 18.60042993, 17.26690807 ,13.20934876,11.68329037,11.02175704,9.851577763])
theta_rel = abs(theta-theta_lit)/theta_lit
print(f"relative abweichungen theta_rel: {theta_rel}")