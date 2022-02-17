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

 
def SFR_law(mat_SF, gas, total, k=None, model_type='cte'):
    '''
    Compute the SFR (sigma_psi) for two different laws: constant time or variable,
    according to the law to be explained in Martínez-Sebastián et al. (in preparation) 
    
    Parameters
    ----------´
    mat_SF : float, 
        Density of the materials considered to form stars.
    gas : float,
        Total density of all gas.
    total : float, 
        Total density of all barionic material.
    k : float, optional
        Unified constant scaling the ISFR. Default is None. 
    model_type : str, optional
        Parameter to select between both models. The options are 'variable'
        or 'cte'. The default is 'cte'.

    Returns
    -------
    None.

    '''
    if model_type=='variable':
        '''
        tau_SF *= total**(-2/3)
        tau_SF *= gas**(-1/3)
        tau_SF *= max(mat, 1e-4)**(1/3)     
        '''
        tau_SF = total**(-1/2)
        tau_SF *= gas**(-1)
        tau_SF *= max(mat_SF, 1e-4)**(1/2)/k        
    elif model_type=='cte':
        tau_SF = 3         #Constant time scale.
#    print(mat, gas,total, tau_SF)
    return mat_SF/tau_SF

#include_atom = .3
#include_mol = 1
class Model:
        '''
        Class to find the observables/state of the model in an iteration. 
        '''
        def equations(self, current_time, current_state):#, model_type='cte',\
            '''Differential quations of the model'''
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
            mat = self.include_atom*atom + self.include_mol*mol
            Psi = SFR_law(mat, gas, total, self.K, self.model_type)
            
            eta_ion_eff = self.eta_ion * (1 - np.exp(-atom/8e-4))
            diss_cross_section_H2 = 7000
            diss_cross_section_dust = 1e5
            diss_optical_depth = mol*(diss_cross_section_H2+Z*diss_cross_section_dust) # Optical depth associated to dissociation process
            #eta_diss_eff = self.eta_dis * (1 - np.exp(-self.Z_sun*diss_cross_section_H2))#8e-3
            eta_diss_eff = self.eta_dis #COMMENT
            eta_diss_eff *= (1 - np.exp(-diss_optical_depth)) #  Photons that arrives
            eta_diss_eff *= diss_cross_section_H2/(diss_cross_section_H2+Z*diss_cross_section_dust) # Photons that actually dissociate molecules.
            didt = self.I_0*np.exp(-current_time/self.tau_I) - recombination \
                + ((1-self.enriched_wind)*self.R
                   + eta_ion_eff - self.wind*ion/gas) * Psi
            dadt = recombination - cloud_formation \
                + (eta_diss_eff - eta_ion_eff - self.wind*atom/gas  - atom/mat*self.include_atom) * Psi
            dmdt = cloud_formation - (mol/mat*self.include_mol + eta_diss_eff + self.wind*R_mol) * Psi
            dsdt = (1-self.R)*Psi
            dodt = ((1-self.enriched_wind)*self.Z_R*self.R - (1+self.wind)*Z) * Psi      
            return [didt, dadt, dmdt, dsdt, dodt]
            
        def __init__(self,
                     model_type,
                     include_atom = 0.0,
                     include_mol = 1.,
                     Sigma_I=100.,  # Msun/pc^2                                 
                     tau_I=3.,  # Gyr
                     T_warm=1.,  # 1e4 K                                        
                     T_cold=1.,  # 100 K
                     Z_eff=6e-6,
                     tau_SF=3.,  # Gyr
                     T1_Twarm=1., # 1
                     eta_ion=100.,
                     eta_dis=50., #100 for variable model and 10 for cte.
                     R=0.17,
                     Z_R=0.06, # 10*Z_sun
                     wind=3, # SN. Mass-loaging factor
                     enriched_wind=0,
                     today=13.7,  # Gyr
                     Z_sun=0.006, # Using oxygen as proxy
                     K = 3
                     ):
            # Defining properties of the class
            self.include_atom=include_atom
            self.include_mol=include_mol
            self.model_type=model_type
            self.K = K
            self.I_0 = Sigma_I/tau_I/(1-np.exp(-today/tau_I))#v
            self.tau_I = tau_I
            self.tau_warm = 0.074*np.power(T_warm, 1.8)  # Gyr ???
            self.tau_cold = 0.797*np.power(T_cold, 0.5)  # Gyr ???
            self.Z_eff = Z_eff
            self.Z_sun = Z_sun
            self.mu_warm = T1_Twarm/tau_SF#v
            self.mu_cold = self.mu_warm * 100*T_warm/T_cold                     # Unit convertion 
            self.eta_ion = eta_ion
            self.eta_dis = eta_dis
            self.R = R
            self.Z_R = Z_R
            self.wind = wind
            self.enriched_wind = enriched_wind
            
            initial_time = 1e-13*today  # to avoid zeroes in the initial conditions
            initial_mass = self.I_0*initial_time
            IC = np.array([initial_mass/3, initial_mass/3, initial_mass/3, 0., 0.])
            # Integration of the model equations to get the actual observables. 
            self.result = integrate.solve_ivp(                                    
                    self.equations, (initial_time, today), IC, method='BDF')   
#r = Model().result.y[:, -1]
#print(r, np.sum(r[:-1]))

def model_run(S_I, K=None, model_type='cte',include_atom = .0,include_mol = 1, **kwargs):
    '''
    Function to run the model, getting the interesting values.

    Parameters
    ----------
    S_I : array-like,
        Values to evaluate a range of accreted masses.
    K : TYPE
        DESCRIPTION.
    model_type : str, optional
        str to change the star formation law. The default is 'cte'.
    include_atom : float, optional
        Value of atomic gas forming stars. The default is .0.
    include_mol : float, optional
        Efficiency of molecular gas forming stars. The default is 1.
    **kwargs 

    Returns
    -------
    model : dictionary,
        Dictionary containing 'HII', 'stars', 'HI', 'H2', 'gas', 'OH', 'total',
        'SFR'.
    '''
    result = []
    for Sigma_I in S_I:
        m = Model(model_type=model_type,Sigma_I=Sigma_I,include_atom = include_atom,include_mol=include_mol, K = K,**kwargs)
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

# <codecell> Constants

G_SI = 6.674e-11 # m^3/kg/s^2
G = G_SI*2e30/3.09e16**3 # pc^3/M_sun/s^2




# <codecell> Bye
# -----------------------------------------------------------------------------
#                                                           ... Paranoy@ Rulz!
# -----------------------------------------------------------------------------