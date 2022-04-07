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

# Constant defintions
G_N_m2_kg_2 = 6.674e-11
G_m3_kg_1_s_2 = G_N_m2_kg_2
G_pc2_km_Msun_1_s_1_Gyr_1 = G_m3_kg_1_s_2*1e9*365.25*24*3600*2e30/1e3/(3.0857e+16**2)
kb_J_K_1 = 1.38e-23
mH_kg = 1.67e-27

class Model:
    def SFR_law(self,mat_SF, mat_gas, mat_total):
        if self.model_type=='variable':
            Psi_t = G_pc2_km_Msun_1_s_1_Gyr_1/self.alpha_ff/self.c_cold_km_s_1*mat_SF*mat_total
            Psi_nt = ((G_pc2_km_Msun_1_s_1_Gyr_1*mat_SF*mat_total/(self.alpha_ff*self.c_cold_km_s_1))**2*mat_gas/self.Tnt_Gyr)**(1/3)
            ret_Msun_Gyr_1 = Psi_t*Psi_nt/np.sqrt(Psi_t**2+Psi_nt**2)
            tau_SF = mat_SF / ret_Msun_Gyr_1#max(mat_SF / ret_Msun_Gyr_1,1)
        elif self.model_type=='cte':
            tau_SF = self.T_SF_Gyr        #Constant time scale.    
        ret_Msun_Gyr_1 = mat_SF/tau_SF
        return ret_Msun_Gyr_1

    def equations(self, current_time, current_state):#, model_type='cte',\
        '''Differential quations of the model'''
        ion, atom, mol, stars, metals = current_state
        
        if mol<0 :
            mol = 0
        gas = ion + atom + mol
        total = ion + atom + mol +stars + metals
        R_mol = mol/gas
        Z = metals/gas
        mat = self.include_atom*atom + self.include_mol*mol
        Psi = self.SFR_law(mat, gas, total)
        pressure = gas * (gas + stars)
        
        '''
        tau_SF = mat/Psi
        self.mu_warm = self.T1_Twarm/tau_SF                                     # Revisar. 
        self.mu_cold = self.mu_warm * self.T_warm/self.T_cold
        recombination = ion/self.tau_warm * pressure/(1+self.mu_warm*R_mol)  
        cloud_formation = (
                atom/self.tau_cold * pressure/(1+self.mu_cold*R_mol)
                * (Z + self.Z_eff)/self.Z_sun
                )
        '''
        recombination = ion/2e-2 * pressure/1e2
        cloud_formation = atom/0.03 * Z/self.Z_sun #* pressure/5000
        eta_ion_eff = self.eta_ion * (1 - np.exp(-atom/8e-4))
        diss_cross_section_H2 = 7000
        diss_cross_section_dust = 1e5 # 1e5 Original used value
        diss_optical_depth = mol*(diss_cross_section_H2+Z*diss_cross_section_dust) # Optical depth associated to dissociation process
        #eta_diss_eff = self.eta_dis * (1 - np.exp(-self.Z_sun*diss_cross_section_H2))#8e-3
        eta_diss_eff = self.eta_dis 
        eta_diss_eff *= (1 - np.exp(-diss_optical_depth)) #  Photons that arrives
        eta_diss_eff *= diss_cross_section_H2/(diss_cross_section_H2+Z*diss_cross_section_dust) # Photons that actually dissociate molecules.

        external_diss = 0
        external_diss *= (1 - np.exp(-diss_optical_depth)) #  Photons that arrives
        external_diss *= diss_cross_section_H2/(diss_cross_section_H2+Z*diss_cross_section_dust) # Photons that actually dissociate molecules.        
        #eta_diss_eff = 0
        didt = self.I_0*np.exp(-current_time/self.tau_I) - recombination \
            + ((1-self.enriched_wind)*self.R
               + eta_ion_eff - self.wind*ion/gas) * Psi
        dadt = recombination - cloud_formation + external_diss \
            + (eta_diss_eff - eta_ion_eff - self.wind*atom/gas  - atom/mat*self.include_atom) * Psi
        dmdt = cloud_formation - external_diss \
            - (mol/mat*self.include_mol + eta_diss_eff + self.wind*R_mol) * Psi #
        dsdt = (1-self.R)*Psi
        dodt = ((1-self.enriched_wind)*self.Z_R*self.R - (1+self.wind)*Z) * Psi   
        
        return [didt, dadt, dmdt, dsdt, dodt]
    
    def __init__(self,
                model_type = 'variable',
                TauSF_ = 1e-14,
                include_atom = 0.0, 
                include_mol = 1.,
                accreted_mass_Msun_pc2=100.,                                 
                infall_time_Gyr=3.,
                T_warm_K=1e4,                                     
                T_cold_K=1e2,  
                Z_eff=6e-6,
                T_SF_Gyr=3.,
                eta_ion=100.,
                eta_dis=50., #100 for variable model and 10 for cte.
                R=0.17,
                Z_R=0.06, # 10*Z_sun
                wind=3, # SN. Mass-loaging factor
                enriched_wind=0,
                today_Gyr=13.7,
                Z_sun=0.006, # Using oxygen as proxy
                ssfr=1,
                alpha_ff = 100,
                Tnt_Gyr =42,
                 ):
        # Defining properties of the class
        self.include_atom=include_atom
        self.include_mol=include_mol
        self.model_type=model_type
        self.alpha_ff = alpha_ff
        self.Tnt_Gyr = Tnt_Gyr
        self.today = today_Gyr
        self.tau_I = infall_time_Gyr
        self.I_0 = (accreted_mass_Msun_pc2
                    / self.tau_I / (1-np.exp(-self.today/self.tau_I)))
        self.T_warm = T_warm_K
        self.T_cold = T_cold_K
        self.tau_warm = 0.074 * np.power(self.T_warm/(1e4), 1.8)
        self.tau_cold = 0.797 * np.power(self.T_cold/(1e2), 0.5)  *10
        self.Z_eff = Z_eff
        self.Z_sun = Z_sun
        c_cold_m_s_1 = np.sqrt(kb_J_K_1*self.T_cold/mH_kg)
        self.c_cold_km_s_1 = c_cold_m_s_1*1e-3
        self.eta_ion = eta_ion
        self.eta_dis = eta_dis
        self.R = R
        self.Z_R = Z_R
        self.wind = wind
        self.enriched_wind = enriched_wind
        self.T_SF_Gyr = T_SF_Gyr
        initial_time_Gyr = 1e-13*today_Gyr  # to avoid zeroes in the initial conditions
        initial_mass = self.I_0*initial_time_Gyr
        IC = np.array([initial_mass/3, initial_mass/3, initial_mass/3, 0., 0.])
        
        # Integration of the model equations to get the actual observables. 
        self.result = integrate.solve_ivp(                                    
                self.equations, (initial_time_Gyr, self.today), IC, method='BDF')      
        mat = self.include_atom*self.result.y[1, -1] + self.include_mol*self.result.y[2, -1]
        gas = self.result.y[0, -1] + self.result.y[1, -1] + self.result.y[2, -1]    
        total = gas + self.result.y[3, -1] + self.result.y[4, -1]
        self.Psi = self.SFR_law(mat, gas, total)
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
    SF_density = []
    '''
    fig,[[ax1,ax2,ax3],[ax4,ax5,ax6]] = plt.subplots(2,3)
    #ax1.set_ylabel(r'SFR history [Gyr$^{-1}$]'),ax2.set_xlabel('Time [Gyr]'),
    ax1.grid(),ax2.grid(),ax3.grid(),ax4.grid(),ax5.grid(),ax6.grid()
    #ax2.set_ylabel('Atomic hydrogen'),ax2.set_ylabel('molecular hydrogen')
    ax1.set_yscale('log'), ax2.set_yscale('log'), ax3.set_yscale('log')
    ax4.set_yscale('log'), ax5.set_yscale('log'), ax6.set_yscale('log')
    ax1.set_ylim(1e-3,1e3), ax2.set_ylim(1e-2,1e3), ax3.set_ylim(1e-2,1e3)
    ax4.set_ylim(1e-2,1e4), ax5.set_ylim(1e-8,1e-2), ax6.set_ylim(1e-2,1e4)
    ax6.axhline(43)
    '''
    for Sigma_I in S_I:
        m = Model(accreted_mass_Msun_pc2=Sigma_I,**kwargs)
        result.append(m.result.y[:, -1])    # -1 last element.   
        
        SF_density.append(m.Psi)
        '''
        #SFR = [SFR_law(mat, gas,tot, model_type=model_type, alpha_ff=alpha_ff, \
        #    Tnt_Gyr=Tnt,  T_SF_Gyr= T_SF_Gyr) for mat, gas, tot in zip(
        #    (include_mol*m.result.y[2,:] + include_atom*m.result.y[1,:]),
        #    m.result.y[0,:]+m.result.y[1,:]+m.result.y[2,:], m.result.y[0,:]+m.result.y[1,:]+m.result.y[2,:]+m.result.y[3,:]+m.result.y[4,:])]
        #ax1.plot(m.result.t,SFR)
        #ax1.text(m.result.t[3], max(SFR), Sigma_I)
        ax2.plot(m.result.t,m.result.y[1,:]) # HI
        ax3.plot(m.result.t,m.result.y[2,:]) # H2
        ax4.plot(m.result.t,m.result.y[3,:]) # stars
        O,gas = m.result.y[4,:],m.result.y[0,:] + m.result.y[1,:] + m.result.y[2,:]
        ax5.plot(m.result.t,O/gas) # metallicity 
        #ax6.plot(m.result.t, (m.result.y[1,:] + m.result.y[2,:])/SFR) # dep. time
        '''
    result = np.array(result)
    model = {}
    model['HII'] = result[:, 0]
    model['stars'] = result[:, 3]
    model['HI'] = result[:, 1]
    model['H2'] = result[:, 2]
    model['gas'] = model['HI'] + model['H2'] + model['HII']
    model['O'] = result[:, 4]     
    model['OH'] = 12 + np.log10(model['O']/16/model['gas'])       
    model['total'] = model['gas'] + model['stars'] + model['O']
    model['SFR'] = SF_density
    return model



# <codecell> Test scaling

#plt.close('all')
if __name__ == "__main__":
    S_I = np.logspace(0, 4, 20)
    kwargs = {'wind': 1.}   
    m = model_run(S_I, infall_time_Gyr=4,\
        model_type='variable', alpha_ff = 100, Tnt_Gyr = 42)
    plt.figure()
    plt.plot(m['stars'], m['SFR'], 'k-',label='SFR')
    #plt.plot(m['stars'], m['H2']/m['HI'], 'r-')
    plt.plot(m['stars'],m['HI'],'r',label=r'$\Sigma_{HI}$')
    plt.plot(m['stars'],m['H2'],'b',label=r'$\Sigma_{H_2}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.xlabel(r'$\Sigma_{\star}[M_{\odot}/pc^2]$')
    plt.show()
    plt.xlim(xmin=1)
    plt.ylim(1e-1,1e3)


# <codecell> Bye
# -----------------------------------------------------------------------------
#                                                           ... Paranoy@ Rulz!
# -----------------------------------------------------------------------------
