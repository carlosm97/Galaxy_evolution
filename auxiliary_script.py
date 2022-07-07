#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:08:59 2022

@author: carlos
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import astropy.io.fits as fits
from astropy.io import ascii
from astropy.table import Table
import os
os.chdir('/home/carlos/Desktop/Paper_TFG/TFG2')
import model_gas as models
                                                                             
if '__file__' in globals():
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
else:
    base_dir = os.path.join('..', '..')  # assume this is the working directory
# <codecell> Read data

# CALIFA (Sánchez et al. ???)
#plt.close('all')
CALIFA_dir = os.path.join('./input/CALIFA')
CALIFA_log_Sigma_Mass_stars = []
CALIFA_log_Sigma_SFR = []
CALIFA_log_Sigma_Mass_gas = []
CALIFA_OH_O3N2 = []
for filename in os.listdir(CALIFA_dir):
    try:
        with fits.open(os.path.join(CALIFA_dir, filename)) as hdu:
            CALIFA_log_Sigma_Mass_stars.append(hdu[0].data[0, 1, :])  # Msun/pc^2
            CALIFA_log_Sigma_SFR.append(hdu[0].data[1, 1, :])  # Msun/pc^2/yr
            CALIFA_log_Sigma_Mass_gas.append(hdu[0].data[2, 1, :])  # Msun/pc^2
            CALIFA_OH_O3N2.append(hdu[0].data[9, 1, :])  # 12 + log(O/H)
    except OSError:
        print('Ignoring', filename)

CALIFA = {}
y = np.array(CALIFA_log_Sigma_Mass_stars)
y[y == 0.] = np.NaN   # It removes 0. as NaN
CALIFA['stars'] = 10**y                                                    
y = np.array(CALIFA_log_Sigma_Mass_gas)
#y[y == 0.] = np.NaN
y *= np.NaN
CALIFA['gas'] = 10**y
y = np.array(CALIFA_log_Sigma_SFR)
y[y == 0.] = np.NaN
CALIFA['SFR'] = 10**(y+9)  # Msun/pc^2/Gyr                                  # yr**-1 to Gyr**-1 
y = np.array(CALIFA_OH_O3N2)
y[y == 0.] = np.NaN
CALIFA['OH'] = y                                                           
bad_OH = np.where(np.isnan(y))
CALIFA['HI'] = np.empty_like(y)*np.NaN                                       
CALIFA['H2'] = np.empty_like(y)*np.NaN


# THINGS (Leroy et al. 2008) + OH (Kudritzki et al. 2015)

Z0, dZdR, R_L08, HI_L08, H2_L08, Stars_L08, SFR_L08 = \
    np.loadtxt(os.path.join('./input/L+08_K+15.txt'),#base_dir, 'input', 'L+08_K+15.txt'),
        usecols=(1, 2, 3, 5, 7, 9, 11), unpack=True)
    
THINGS = {}
THINGS['stars'] = Stars_L08
THINGS['HI'] = HI_L08
THINGS['H2'] = H2_L08
THINGS['gas'] = HI_L08 + H2_L08
THINGS['SFR'] = 0.1*SFR_L08  # Msun/pc^2/Gyr                                
THINGS['OH'] = Z0 - dZdR*R_L08

#%% Adding sound speed when possible.  

gal_T, Z0_str, dZdR_str, R_L08_str, r_25_str, c5, c6, c7, c8, star_T_str, c10, c11, c12 = \
    np.genfromtxt(os.path.join('./input/L+08_K+15.txt'),
        usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),dtype=str, unpack=True)
r_25 = [float(j) for j in r_25_str]

T09_galaxies = os.listdir('./input/Tamburro+09/txt/sigma')

def find_nearests(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def interp_sig(galaxy,r):
    R, sigma = np.loadtxt(os.path.join('./input/Tamburro+09/txt/sigma/'+galaxy), unpack=True)
    index = find_nearests(R, r)
    return np.polyval(np.polyfit([R[index],R[index+1]],[sigma[index],sigma[index+1]],1),r)

interpolated_dispersion = np.zeros(len(gal_T))
for i in range(len(gal_T)): 
    if gal_T[i]+'.txt' in T09_galaxies:
        interpolated_dispersion[i]=round(interp_sig(gal_T[i]+'.txt',r_25[i]),2)
    
data = Table()
data['# Gal'], data['Z0'], data['dZ/dR'], data['R'], data['Rnorm'], data['Sigma_HI'], data['dSigma_HI'],\
data['Sigma_H2'], data['dSigma_H2'], data['Sigma_*'], data['dSigma_*'], data['Sigma_SFR'], data['dSigma_SFR'] =\
gal_T, c1, c2, c3, r_25_str, c5, c6, c7, c8, star_T, c10, c11, c12
data['velocity_dispersion'] = interpolated_dispersion
  
ascii.write(data, './input/L+08_T+09_K+15.txt', overwrite=True)

#%% Comparing THINGS & CALIFA metallicities

CALIFA_galaxies = os.listdir('./input/CALIFA')

CALIFA_dir = os.path.join('./input/CALIFA')

CALIFA_log_Sigma_SFR = []
CALIFA_log_Sigma_Mass_gas = []
CALIFA_OH_O3N2 = []

common_files = []
for gal in set(gal_T):
    for filename in os.listdir(CALIFA_dir):
        if gal in filename:
            common_files.append(filename)
            
'''
fig, axs = plt.subplots(len(common_files))
for filenumber in range(len(common_files)):
    hdu = fits.open(os.path.join('./input/CALIFA', common_files[filenumber]))
    x = np.array(hdu[0].data[0, 1, :])
    x[x == 0.] = np.nan
    y = hdu[0].data[9, 1, :]
    y[y == 0] = np.nan
    axs[int(filenumber)].set_xscale('log')
    axs[int(filenumber)].plot(x,y,'.')
plt.show()
'''
fig, axs = plt.subplots(2)

for filename in np.sort(common_files):
    hdu = fits.open(os.path.join('./input/CALIFA', filename))
    x = np.array(hdu[0].data[0, 1, :])
    x[x == 0.] = np.nan
    x = 10**x
    y = hdu[0].data[9, 1, :]
    y[y == 0] = np.nan
    if 'NGC6946' in filename:
        axs[0].plot(x,y,'*',label=filename[7:9])
    else:
        axs[1].plot(x,y,'*')
    axs[0].set_xscale('log'), axs[1].set_xscale('log')
    
plt.show()

Z0, dZdR, R_L08, star_T = Z0_str.astype(float), dZdR_str.astype(float), R_L08_str.astype(float), star_T_str.astype(float)

OH_T = Z0 - dZdR*R_L08

axs[0].plot(star_T[gal_T == 'NGC6946'], OH_T[gal_T == 'NGC6946'],'s')
axs[1].plot(star_T[gal_T == 'NGC4736'], OH_T[gal_T == 'NGC4736'],'s')

axs[0].legend()
axs[0].grid(), axs[1].grid()
fig.supxlabel(r'$\Sigma_* [M_{\odot}/pc^2]$'), fig.supylabel(r'12 + log(O/H)')

axs[0].set_title('NGC6946'), axs[1].set_title('NGC4736')
plt.savefig('metallicity_comparison.pdf')





