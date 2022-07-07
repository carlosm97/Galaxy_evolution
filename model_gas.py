#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Second implementation of chemical evolution model

@author: carlos
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

# Constant defintions
G_N_m2_kg_2 = 6.674e-11
G_m3_kg_1_s_2 = G_N_m2_kg_2
G_pc3_Msun_1_s_2 = G_m3_kg_1_s_2*2e30/(3.0857e+16**3)
#G_pc4_m_1_kg_Msun_2_s_2 = G_pc3_Msun_1_s_2*2e30/3.0857e+16
#G_pc2_km_Msun_1_s_1_Gyr_1 = G_pc3_Msun_1_s_2*1e9*365.25*24*3600*3.0857e+16/1e3
G_pc3_Msun_1_Gyr_2 = G_pc3_Msun_1_s_2*(1e9*365.25*24*3600)**2
kb_J_K_1 = 1.38e-23
mH_kg = 1.67e-27

conversion_Msun_pc_2_to_uma_cm_2 = 1.26e20

class Model:
    def SFR_function(self,current_state):
        gas, stars = current_state
        
        total = gas + stars
        
        self.P = np.pi/2 * G_pc3_Msun_1_Gyr_2 * gas * total 
        #factor = (self.P/1e7)**0.25
        #factor = (gas/total)**-0.9
        factor = 5*(gas/total)**-0.25
        return self.P/(self.nu*factor)
    
    def equations(self, current_time, current_state):#, model_type='cte',\
        '''Differential quations of the model'''
        
        self.SFR = self.SFR_function(current_state)
        
        dgdt = self.I_0*np.exp(-current_time/self.tau_I) - (self.w + 1) * self.SFR
        dsdt = self.SFR
        
        return [dgdt, dsdt]
    def __init__(self,
                accreted_mass_Msun_pc2=100.,
                infall_time_Gyr = 3.,
                tau_SF = 1.,
                tau_SN = 1.,
                E_SN_erg = 1e51, # erg
                R_SN = 7e-3, # SN/M_sun
                R = 0.17,
                wind = 0.,
                today_Gyr = 13.7,
                 ):
        self.R = R
        self.w = wind/(1-self.R)
        #self.t_ff = t_ff_Gyr
        # self.t_SN = t_SN_Gyr
        self.E_SN = E_SN_erg*1e-7*(1e9*365.25*24*3600)**2/(2e30 * 3.0857e+16**2)
        self.R_SN = R_SN
        self.tau_SF = tau_SF
        self.tau_SN = tau_SN 
        self.nu = np.sqrt(self.tau_SF * self.R_SN * self.E_SN * self.tau_SN)/(1 - self.R)
        self.I_0 = accreted_mass_Msun_pc2
        self.tau_I = infall_time_Gyr
        self.today = today_Gyr
        initial_time_Gyr = 1e-13*today_Gyr  # to avoid zeroes in the initial conditions
        initial_mass = self.I_0*initial_time_Gyr
        IC = np.array([initial_mass, 0.])
        
        # Integration of the model equations to get the actual observables. 
        self.result = integrate.solve_ivp(                                    
                self.equations, (initial_time_Gyr, self.today), IC, method='BDF')      

def model_run(S_I, **kwargs):
    '''
    Function to run the model, getting the interesting values.

    Parameters
    ----------
    S_I : array-like,
        Values to evaluate a range of accreted masses.
    **kwargs 

    Returns
    -------
    model : dictionary,
        Dictionary containing 'HII', 'stars', 'HI', 'H2', 'gas', 'OH', 'total',
        'SFR'.
    '''
    result = []
    P, SFR = [], []

    for Sigma_I in S_I:
        m = Model(accreted_mass_Msun_pc2=Sigma_I,**kwargs)
        result.append(m.result.y[:, -1])    # -1 last element.  
        P.append(m.P), SFR.append(m.SFR)

    result = np.array(result)
    model = {}
    model['gas'] = result[:, 0]
    model['stars'] = result[:, 1]   
    model['total'] = model['gas'] + model['stars']
    model['P'] = P
    model['SFR'] = SFR
    model['cs'] = 591*(model['total']/model['gas'] )**0.25
    model['velocity_HI'] = 591/80*(model['total']/model['gas'] )**0.25
    return model



# <codecell> Test scaling

#plt.close('all')
if __name__ == "__main__":
    S_I = np.logspace(0, 4, 20)
    kwargs = {'wind': 0.}   
    #m = model_run(S_I, infall_time_Gyr=4, model_type='variable', include_atom=1, eta_dis=0, **kwargs)
    m = model_run(S_I, infall_time_Gyr=1e-3,**kwargs)
    plt.figure()
    plt.plot(m['stars'], (m['total']/m['gas'])**0.25, 'k-',label='SFR')
    #plt.plot(m['stars'], m['H2']/m['HI'], 'r-')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.xlabel(r'$\Sigma_{\star}[M_{\odot}/pc^2]$')
    plt.show()

# <codecell> Bye
# -----------------------------------------------------------------------------
#                                                           ... Paranoy@ Rulz!
# -----------------------------------------------------------------------------
