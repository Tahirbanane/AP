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

h=const.h
c=const.c
d_LiF=201.4*10**(-12) #m
ry=13.6 #eV


print (f'\n -----------------------------------Bragg-Bedingung-----------------------------------')

theta_bragg,N_bragg=np.genfromtxt('bragg.txt', unpack = True)
plt.plot(theta_bragg, N_bragg,'kx',label = 'Messdaten')
plt.ylabel(f"Impulse pro Sekunde")
N_max = find_peaks(N_bragg, height=100)
N_peak_bragg = N_bragg[N_max[0]]
theta_peak_bragg = theta_bragg[N_max[0]]
plt.plot(theta_peak_bragg,N_peak_bragg,'r.',label = 'Maximum')
plt.xlabel(f"Winkel / °")
plt.legend()
plt.savefig("bragg.pdf")
plt.close()

print(f"maximale impulsrate von {N_peak_bragg} bei {theta_peak_bragg}°")

print (f'\n -----------------------------------Emissionsspektrum-----------------------------------')

theta_cu,N_cu=np.genfromtxt('emissionsspektrum.txt', unpack = True)
plt.plot(theta_cu, N_cu,'kx',label = 'Messdaten')
plt.ylabel(f"Impulse pro Sekunde")
N_max = find_peaks(N_cu, height=1000)
N_peak_cu = N_cu[N_max[0]]
theta_peak_cu = theta_cu[N_max[0]]
plt.axvline(theta_peak_cu[0], color='steelblue', ls = '--', label = r'$K_{\mathrm{\beta}}$')
plt.axvline(theta_peak_cu[1], color='g', ls = '--', label = r'$K_{\mathrm{\alpha}}$')
plt.xlabel(f"Winkel / °")
plt.legend()
plt.savefig("emissionsspektrum.pdf")
plt.close()
#bremsberg einfügen

print(f"k-linien bei {theta_peak_cu}°")

#grenzwinkel fehlt

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
plt.ylabel(f"Impulse pro Sekunde")
plt.xlabel(f"Winkel / °")
plt.legend()
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

z=29
E_K_lit =  8980.476 #eV
sigma_1 = z - np.sqrt(E_K_lit/ry) #°
sigma_2 = z - np.sqrt(4*(z-sigma_1)**2 - (4/ry)*E_K_peaka) #°
sigma_3 = z - np.sqrt(9*(z-sigma_1)**2 - (9/ry)*E_K_peakb) #°

print(f"sigma_1 = {sigma_1} , sigma_2 = {sigma_2} , sigma_3 = {sigma_3}")
print(f"E_K_alpha = {E_K_peaka}eV, E_K_beta = {E_K_peakb}eV ")
print(f"E_FWHM_alpha = {FWHMa}eV, E_FWHM_beta = {FWHMb}eV")
print(f"A_alpha = {Aa} A_beta = {Ab}")

print (f'\n -----------------------------------Absorptionsspektrum-----------------------------------')

#zink
theta_zn,N_zn=np.genfromtxt('zink.txt', unpack = True)
plt.plot(theta_zn, N_zn,'kx',label = 'Messdaten')
plt.axvline(18.5, color='steelblue', ls = '--', label = r'$N_{min}$')
plt.axvline(19.0, color='g', ls = '--', label = r'$N_{max}$')
plt.ylabel(f"Impulse pro Sekunde")
plt.xlabel(f"Winkel / °")
plt.legend()
plt.savefig("zink.pdf")
plt.close()
N_K_zn = N_zn[5]+(N_zn[10]-N_zn[5])/2
#E_K_zn

#brom
theta_br,N_br=np.genfromtxt('brom.txt', unpack = True)
plt.plot(theta_br, N_br,'kx',label = 'Messdaten')
plt.axvline(13.0, color='steelblue', ls = '--', label = r'$N_{min}$')
plt.axvline(13.5, color='g', ls = '--', label = r'$N_{max}$')
plt.ylabel(f"Impulse pro Sekunde")
plt.xlabel(f"Winkel / °")
plt.legend()
plt.savefig("brom.pdf")
plt.close()
N_K_br = N_br[2]+(N_br[7]-N_br[2])/2

#gallium
theta_ga,N_ga=np.genfromtxt('gallium.txt', unpack = True)
plt.plot(theta_ga, N_ga,'kx',label = 'Messdaten')
plt.axvline(17.1, color='steelblue', ls = '--', label = r'$N_{min}$')
plt.axvline(17.6, color='g', ls = '--', label = r'$N_{max}$')
plt.ylabel(f"Impulse pro Sekunde")
plt.xlabel(f"Winkel / °")
plt.legend()
plt.savefig("gallium.pdf")
plt.close()
N_K_ga = N_ga[1]+(N_ga[6]-N_ga[1])/2

#rubidium
theta_rb,N_rb=np.genfromtxt('rubidium.txt', unpack = True)
plt.plot(theta_rb, N_rb,'kx',label = 'Messdaten')
plt.axvline(11.5, color='steelblue', ls = '--', label = r'$N_{min}$')
plt.axvline(12.1, color='g', ls = '--', label = r'$N_{max}$')
plt.ylabel(f"Impulse pro Sekunde")
plt.xlabel(f"Winkel / °")
plt.legend()
plt.savefig("rubidium.pdf")
plt.close()
N_K_rb = N_rb[3]+(N_rb[9]-N_rb[3])/2

#strontium
theta_sr,N_sr=np.genfromtxt('strontium.txt', unpack = True)
plt.plot(theta_sr, N_sr,'kx',label = 'Messdaten')
plt.axvline(10.9, color='steelblue', ls = '--', label = r'$N_{min}$')
plt.axvline(11.4, color='g', ls = '--', label = r'$N_{max}$')
plt.ylabel(f"Impulse pro Sekunde")
plt.xlabel(f"Winkel / °")
plt.legend()
plt.savefig("strontium.pdf")
plt.close()
N_K_sr = N_sr[4]+(N_sr[9]-N_sr[4])/2

