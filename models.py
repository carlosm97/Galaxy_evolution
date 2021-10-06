#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New implementation of chemical evolution model

@author: Yago Ascasibar (UAM)
Created on Mon Sep  3 14:19:34 2018
"""
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import os

if '__file__' in globals():
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
else:
    base_dir = os.path.join('..', '..')  # assume this is the working directory
Cons = np.linspace(0, 2, 21)

 
def SFR_law(mat, gas, total, k, model_type='cte'):
    if model_type=='variable':
        tau_SF = k                 # Variable time scale.
        tau_SF *= total**(-2/3)
        tau_SF *= gas**(-1/3)
        tau_SF *= max(mat, 1e-4)**(1/3)     
    elif model_type=='cte':
        tau_SF = 3.0         #Constant time scale.
#    print(mat, gas,total, tau_SF)
    return mat/tau_SF

include_atom = .3
include_mol = 1


class Model:
        def equations(self, current_time, current_state, model_type='cte'):
            ion, atom, mol, stars, metals = current_state
            gas = ion + atom + mol
            pressure = gas * (gas + stars)     
            total = ion+atom+mol+stars+metals
            R_mol = mol/gas
            Z = metals/gas
            recombination = ion/self.tau_warm * pressure/(1+self.mu_warm*R_mol)  
            cloud_formation = (
                    atom/self.tau_cold * pressure/(1+self.mu_cold*R_mol)
                    * (Z + self.Z_eff)/self.Z_sun
                    )
            mat = include_atom*atom + include_mol*mol
            Psi = SFR_law(mat, gas, total, self.K, model_type)
            
            eta_ion_eff = self.eta_ion * (1 - np.exp(-atom/8e-4))
            eta_diss_eff = self.eta_dis * (1 - np.exp(-mol/1.5e-4))#8e-3
            didt = self.I_0*np.exp(-current_time/self.tau_I) - recombination \
                + ((1-self.enriched_wind)*self.R
                   + eta_ion_eff - self.wind*ion/gas) * Psi
            dadt = recombination - cloud_formation \
                + (eta_diss_eff - eta_ion_eff - self.wind*atom/gas  - atom/mat*include_atom) * Psi
            dmdt = cloud_formation - (mol/mat*include_mol + eta_diss_eff + self.wind*R_mol) * Psi
            
            
            dsdt = (1-self.R)*Psi
            dodt = ((1-self.enriched_wind)*self.Z_R*self.R - (1+self.wind)*Z) * Psi      
            return [didt, dadt, dmdt, dsdt, dodt]
            
        def __init__(self,
                     Sigma_I=100.,  # Msun/pc^2                                     
                     tau_I=3.,  # Gyr
                     T_warm=1.,  # 1e4 K                                        
                     T_cold=1.,  # 100 K
                     Z_eff=6e-6,
                     tau_SF=3.,  # Gyr
                     T1_Twarm=5., # 1
                     eta_ion=100.,
                     eta_dis=10., #100 for variable model and 75 for cte.
                     R=0.17,
                     Z_R=0.06,
                     wind=3,
                     enriched_wind=0,
                     today=13.7,  # Gyr
                     Z_sun=0.006,
                     K = 3
                     ):
            self.K = K
            self.I_0 = Sigma_I/tau_I/(1-np.exp(-today/tau_I))#v
            self.tau_I = tau_I
            self.tau_warm = 0.074*np.power(T_warm, 1.8)  # Gyr ???
            self.tau_cold = 0.797*np.power(T_cold, 0.5)  # Gyr ???
            self.Z_eff = Z_eff
            self.Z_sun = Z_sun
            #self.tau_SF = tau_SF
            #self.tau_SF = K*Sigma_gas**(1/3)*Sigma_m**(-1/3)*Sigma_tot**(2/3)
            self.mu_warm = T1_Twarm/tau_SF#v
            self.mu_cold = self.mu_warm * 100*T_warm/T_cold                     # Unit convertion 
            self.eta_ion = eta_ion
            self.eta_dis = eta_dis
            self.R = R
            self.Z_R = Z_R
            self.wind = wind
            self.enriched_wind = enriched_wind
            #G=6.674e-11 # N*m**2/ kg**2
            #k=1.38e-34  # J*K**-1
            initial_time = 1e-13*today  # to avoid zeroes in the initial conditions
            initial_mass = self.I_0*initial_time
            IC = np.array([initial_mass/3, initial_mass/3, initial_mass/3, 0., 0.])
    
            self.result = integrate.solve_ivp(                                    
                    self.equations, (initial_time, today), IC, method='BDF')   
#r = Model().result.y[:, -1]
#print(r, np.sum(r[:-1]))

def model_run(S_I, K, model_type='cte', **kwargs):
    result = []
    for Sigma_I in S_I:
        m = Model(Sigma_I=Sigma_I, K = K, **kwargs)
        result.append(m.result.y[:, -1])    # -1 last element.   
    result = np.array(result)
        
    model = {}
    model['HII'] = result[:, 0]
    model['stars'] = result[:, 3]
    model['HI'] = result[:, 1]
    model['H2'] = result[:, 2]
    model['gas'] = model['HI'] + model['H2'] + model['HII']             
    model['OH'] = 12 + np.log10(result[:, 4]/16/model['gas'])       
    model['total'] = model['gas'] + model['stars'] + result[:, 4]
    model['SFR'] = [SFR_law(mat, gas,tot, K, model_type) for mat, gas, tot in zip(
            (include_mol*model['H2'] + include_atom*model['HI']),
            model['gas'], model['total'])] #+model['HI']
    return model

# <codecell> Test scaling


if __name__ == "__main__":

    S_I = np.logspace(0, 4, 50)
    Co = 70
    for include_atom in Cons:  
        #print(tau_I)
        plt.figure()
        for tau_I in [1e-3, 2, 4,1e3]:
            m = model_run(S_I, K=300, tau_I=tau_I)
            #print(m)
            plt.plot(m['stars'], m['SFR'], 'k-')
            plt.plot(m['stars'], m['gas']/m['HII'], 'r-')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


# <codecell> Bye
# -----------------------------------------------------------------------------
#                                                           ... Paranoy@ Rulz!
# -----------------------------------------------------------------------------