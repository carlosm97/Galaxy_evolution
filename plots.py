from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import astropy.io.fits as fits
import os
os.chdir('/home/carlos/Desktop/TFG2')
import models
                                                                             
if '__file__' in globals():
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
else:
    base_dir = os.path.join('..', '..')  # assume this is the working directory


# <codecell> Read data

# CALIFA (SÃ¡nchez et al. ???)
plt.close('all')
CALIFA_dir = os.path.join('./input/CALIFA')#base_dir, 'input', 'CALIFA')
CALIFA_log_Sigma_Mass_stars = []
CALIFA_log_Sigma_SFR = []
CALIFA_log_Sigma_Mass_gas = []
CALIFA_OH_O3N2 = []
for filename in os.listdir(CALIFA_dir):
    try:
        with fits.open(os.path.join(CALIFA_dir, filename)) as hdu:
            # print(filename)
            # hdu.info()
            CALIFA_log_Sigma_Mass_stars.append(hdu[0].data[0, 1, :])  # Msun/pc^2
            CALIFA_log_Sigma_SFR.append(hdu[0].data[1, 1, :])  # Msun/pc^2/yr
            CALIFA_log_Sigma_Mass_gas.append(hdu[0].data[2, 1, :])  # Msun/pc^2
#            CALIFA_log_Sigma_Mass_gas_ssp.append(hdu[0].data[3, 1, :])  # ???
#            CALIFA_OH_t2.append(hdu[0].data[8, 1, :])  # 12 + log(O/H)
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
        usecols=(0, 1, 2, 4, 6, 8, 10), unpack=True)
THINGS = {}
THINGS['stars'] = Stars_L08
THINGS['HI'] = HI_L08
THINGS['H2'] = H2_L08
THINGS['gas'] = HI_L08 + H2_L08
THINGS['SFR'] = 0.1*SFR_L08  # Msun/pc^2/Gyr                                
THINGS['OH'] = Z0 - dZdR*R_L08


# Milky Way (MollÃ¡ et al. 2015)

MW = {}
#MW_R    = np.array([     0,     1,     2,    3,    4,    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,   15,   16,     17,    18,    19,    20])
MW['stars'] = 10**np.array([np.NaN, np.NaN, np.NaN, 2.43, 2.50, 2.40, 2.25, 2.09, 1.95, 1.79, 1.69, 1.51, 1.38, 1.25, 1.09, 0.94, 0.80, np.NaN, np.NaN, np.NaN, np.NaN])
MW['HI'] = np.array([9.41, 3.97, 2.37, 2.39, 3.86, 5.06, 5.04, 5.44, 5.69, 7.69, 6.52, 6.16, 5.63, 4.83, 3.65, 2.96, 2.42, 2.15, 1.61, 1.18, 1.1])
MW['H2'] = np.array([0.30, 3.82, 5.18, 3.48, 5.69, 8.28, 8.47, 4.59, 3.15, 2.44, 1.96, 1.24, 0.99, 0.57, 0.82, 1.09, .20, .13, .08, .03, np.NaN])
MW['gas'] = MW['HI'] + MW['H2']
MW['SFR'] = 10**np.array([-.37, .603, .706, .983, 1.163, 1.185, 1.181, .963, .723, .594,  .51, .403, .006, .183, -.26, -.132, -.52, -.68, -.89, -1.37, -1.37])
MW['OH'] = np.array([9.02, 8.86, 8.74, 8.62, 8.82, 8.83, 8.77, 8.69, 8.56, 8.60, 8.45, 8.41, 8.44, 8.44, 8.42, 8.14, 8.14, 8.19, 7.96, np.NaN, np.NaN])


# %% Derived quantities

def compute_derived(*args):
    for dataset in args:
        dataset['P'] = dataset['gas'] * (dataset['gas'] + dataset['stars'])
        dataset['SFE'] = dataset['SFR']/dataset['gas']                      # Overestimation of gas. Maybe related with underestimation of metallicity
        dataset['SSFR'] = dataset['SFR']/dataset['stars']
        dataset['Rmol'] = dataset['H2']/dataset['HI']
        dataset['SFmol'] = dataset['SFR']/dataset['H2']
        dataset['s'] = dataset['stars']/dataset['gas']                      # How many gas is containing in stars. 
        dataset['sfr'] = dataset['SFR']
        
data_list = [CALIFA, THINGS, MW]
compute_derived(*data_list)


# <codecell> Models                                                   
plt.close('all')
class plot_pars:
    """ Settings for each quantity to be plotted """
    def __init__(self, title, xmin, xmax, scale='log',
                 cmin=np.NaN, cmax=np.NaN, cmap='rainbow'):
        self.title = title
        self.xmin = xmin
        self.xmax = xmax
        self.scale = scale
        self.cmap = plt.get_cmap(cmap)
        norm = colors.Normalize if scale == 'linear' else colors.LogNorm
        self.norm = norm(xmin if np.isnan(cmin) else cmin,
                         xmax if np.isnan(cmax) else cmax)


pars = {}
pars['stars'] = plot_pars('$\\Sigma_*$ [M$_\\odot$ pc$^{-2}$]', .3, 3e4)
#pars['stars'] = plot_pars('Densidad superficial de estrellas [M$_\\odot$ pc$^{-2}$]', .3, 3e4)
pars['gas'] = plot_pars('$\\Sigma_{gas}$ [M$_\\odot$ pc$^{-2}$]', 3e-2, 3e3)
pars['SSFR'] = plot_pars('SSFR [Gyr$^{-1}$]', 3e-5, 3)
pars['SFE'] = plot_pars('SFE [Gyr$^{-1}$]', 3e-3, 3e2)
pars['OH'] = plot_pars('12 + log(O/H)', 7.9, 9.1, 'linear', 8.25, 8.55)
#pars['P'] = plot_pars(
#       '$\\Sigma_{gas} (\\Sigma_{gas}+\\Sigma_*)$ [M$_\\odot$ pc$^{-2}$]',
#       3, 3e6)
pars['P'] = plot_pars(
       'Presión [M$_\\odot$ pc$^{-2}$]',
       3, 3e6)
pars['Rmol'] = plot_pars('$\\Sigma_{\\rm H_2}$ / $\\Sigma_{\\rm HI}$', 3e-2, 3e3)
pars['SFmol'] = plot_pars('$\\Psi/\\Sigma_{\\rm H_2}$', 3e-2, 3e1)
pars['sfr'] = plot_pars('SFR[Gyr$^{-1}$]', 3e-2, 3e2)

# Function definitions

def plot_data(x, y, color, ax):
    """ Plot all datasets """
    cbar = pars[color]
    for dataset in data_list:
        xx = dataset[x]
        yy = dataset[y]
        c = dataset[color]
        m = dataset['marker']
        e = dataset['edge']
        s = dataset['size']
        a = dataset['alpha']
        sc = ax.scatter(xx, yy, s=s, c=c, marker=m,
                        edgecolors=e, linewidths=.2,
                        cmap=cbar.cmap, norm=cbar.norm, alpha=a)
        bad_color = np.where(np.isnan(c))
        
        ax.plot(xx[bad_color], yy[bad_color], color='gray', linestyle='None',
                marker=m, ms=s, alpha=a*.1)

    for model in model_list:
        ax.plot(model[x], model[y],
                marker='None', color=model['color'], linestyle=model['style'])

    return sc

def column_plot(x, plots, color, save=True, title=None):
    """ Plot a set of quantities as a function of x, colored by color """
    '''
    # Set dimensions
    n_plots = len(plots)
    panel_width = 7  # inches 5
    #panel_height = panel_width/1.5
    panel_height = panel_width/np.power(n_plots, .3)
    cbar_height = .07*panel_height
    big_margin = .15*panel_width
    small_margin = big_margin/3

    width = big_margin + panel_width + small_margin
    height = (big_margin + n_plots*panel_height  # plots
              + big_margin + cbar_height + big_margin)  # color bar

    fig = plt.figure(figsize=(width, height))

    left = big_margin/width
    bottom = big_margin/height
    panel_width /= width
    panel_height /= height
    cbar_height /= height

    # Plot data

    xmin = pars[x].xmin
    xmax = pars[x].xmax
    xscale = pars[x].scale

    for i, y in enumerate(plots):
        ax = fig.add_axes([left, bottom+i*panel_height,
                           panel_width, panel_height])

        ax.set_xlim(xmin, xmax)
        ax.set_xscale(xscale)
        if i == 0:
            ax.set_xlabel(pars[x].title)
        ax.set_ylim(pars[y].xmin, pars[y].xmax)
        ax.set_yscale(pars[y].scale)
        ax.set_ylabel(pars[y].title)
        ax.tick_params(axis='both', which='both', direction='in',
                       top=True, right=True, labelbottom=(i == 0))
        sc = plot_data(x, y, color, ax)

    # Color bar

    ax_cbar = fig.add_axes([left, bottom+n_plots*panel_height+bottom,
                            panel_width, cbar_height])
    cbar = pars[color]
    bar = plt.colorbar(sc, cmap=cbar.cmap, norm=cbar.norm, orientation='horizontal')
    bar.solids.set_alpha(1)
    ax_cbar.set_title(cbar.title)
#    fig.savefig(os.path.join('./paper/figs/',x+'gas.pdf'))#base_dir, 'paper', 'figs', x+'.pdf'))#################
#    fig.savefig(os.path.join(base_dir, plots[0]+'.png'))
    #plt.close()
    '''
    fs=15
    # Set dimensions

    n_plots = len(plots)
    panel_width = 6  # inches
    #panel_height = panel_width/1.5
    panel_height = panel_width/np.power(n_plots, .5)
    cbar_height = .07*panel_height
    big_margin = .15*panel_width
    small_margin = big_margin/3

    width = big_margin + panel_width + small_margin
    height = (big_margin + n_plots*panel_height  # plots
              + big_margin + cbar_height + big_margin)  # color bar

    fig = plt.figure(figsize=(width, height))

    left = big_margin/width
    bottom = big_margin/height
    panel_width /= width
    panel_height /= height
    cbar_height /= height

    # Plot data

    xmin = pars[x].xmin
    xmax = pars[x].xmax
    xscale = pars[x].scale

    for i, y in enumerate(plots):
        ax = fig.add_axes([left, bottom+i*panel_height,
                           panel_width, panel_height])

        ax.set_xlim(xmin, xmax)
        ax.set_xscale(xscale)
        if i == 0:
            ax.set_xlabel(pars[x].title,fontsize=fs)
        ax.set_ylim(pars[y].xmin, pars[y].xmax)
        ax.set_yscale(pars[y].scale)
        ax.set_ylabel(pars[y].title,fontsize=fs)
        ax.tick_params(axis='both', which='both', direction='in',
                       top=True, right=True, labelbottom=(i == 0),labelsize=fs)
        sc = plot_data(x, y, color, ax)

    # Color bar

    ax_cbar = fig.add_axes([left, bottom+n_plots*panel_height+bottom,
                            panel_width, cbar_height])
    cbar = pars[color]
    bar = plt.colorbar(sc, cax=ax_cbar, cmap=cbar.cmap, norm=cbar.norm, orientation='horizontal')
    bar.solids.set_alpha(1)
    bar.ax.tick_params(labelsize=fs)
    ax_cbar.set_title(cbar.title,fontsize=fs)
    if save==True and title==None:
        fig.savefig(os.path.join('./paper/figs/',x+'Ysfr.pdf'))
    elif save==True and title!=None:
         fig.savefig(os.path.join('./paper/figs/',title+x+'Ysfr.pdf'))       
    #plt.close()
   
    
    
Cons=[100]#np.linspace(400,800,1)# Use the loop to look for the best value of K, with the rest of parameters fixed. 
for CO in Cons: 
    print(CO)
    S_I = np.logspace(0, 4, 10)
    kwargs = {'tau_SF': 1.5, 'wind': 2.}
    tau_0 = models.model_run(S_I, model_type='cte', K=CO,  tau_I=1e-3, include_atom =.2, **kwargs)
    tau_2 = models.model_run(S_I, model_type='cte', K=CO, tau_I=2., include_atom =.2, **kwargs)
    tau_4 = models.model_run(S_I, model_type='cte', K=CO, tau_I=4., include_atom =.2, **kwargs)
    tau_inf = models.model_run(S_I, model_type='cte', K=CO, tau_I=1e3, include_atom =.2, **kwargs)
    
    'Aquí usamos model'
    
    
    model_list = [tau_0, tau_2, tau_4, tau_inf] #[tau_0, 
    compute_derived(*model_list)
    
    CALIFA['marker'] = 'o'
    CALIFA['edge'] = 'face'
    CALIFA['size'] = 5
    CALIFA['alpha'] = 0.2
    
    THINGS['marker'] = 's'
    THINGS['edge'] = 'face'
    THINGS['size'] = 5
    THINGS['alpha'] = 1
    
    MW['marker'] = '*'
    MW['edge'] = 'k'
    MW['size'] = 20
    MW['alpha'] = 1
    
    tau_0['color'] = 'red'
    tau_2['color'] = 'red'
    tau_4['color'] = 'blue'
    tau_inf['color'] = 'blue'
    tau_0['style'] = '-'
    tau_2['style'] = '--'
    tau_4['style'] = '-.'
    tau_inf['style'] = ':'
    
    # Parameters for each quantity
    plot_list = ['SFmol', 'Rmol', 'SFE', 'gas', 'OH', 'SSFR','sfr']#'SFmol', 'Rmol', 'SFE', 'gas', 'OH', 'SSFR','sfr'
    column_plot('stars', plot_list, 'OH',title='_cte_atom_')
    #column_plot('P', plot_list, 'OH',title=)
    #plt.title(CO)

    kwargs = {'wind': 1.5}   
    #tau_0 = models.model_run(S_I,  K=50, include_atom =.5, tau_I=1e-3,model_type='variable', eta_dis=10,**kwargs)
    tau_2 = models.model_run(S_I,  K=100, include_atom =1, tau_I=2.,model_type='variable', eta_dis=100,**kwargs)
    tau_4 = models.model_run(S_I,  K=100, include_atom =1, tau_I=4.,model_type='variable', eta_dis=100,**kwargs)
    tau_inf = models.model_run(S_I, K=100, include_atom =1, tau_I=1e3,model_type='variable', eta_dis=100,**kwargs)
    # K=50.,
    
    #model_list = [tau_0, tau_2, tau_4, tau_inf] #[tau_0, 
    model_list = [tau_2, tau_4, tau_inf] 
    compute_derived(*model_list)
    
    CALIFA['marker'] = 'o'
    CALIFA['edge'] = 'face'
    CALIFA['size'] = 5
    CALIFA['alpha'] = 0.2
    
    THINGS['marker'] = 's'
    THINGS['edge'] = 'face'
    THINGS['size'] = 5
    THINGS['alpha'] = 1
    
    MW['marker'] = '*'
    MW['edge'] = 'k'
    MW['size'] = 20
    MW['alpha'] = 1
    
    #tau_0['color'] = 'red'
    tau_2['color'] = 'red'
    tau_4['color'] = 'blue'
    tau_inf['color'] = 'blue'
    #tau_0['style'] = '-'
    tau_2['style'] = '--'
    tau_4['style'] = '-.'
    tau_inf['style'] = ':'
    
    # Parameters for each quantity
    plot_list = ['SFmol', 'Rmol', 'SFE', 'gas', 'OH', 'SSFR','sfr']
    column_plot('stars', plot_list, 'OH',title='_variable_atom_')
    #column_plot('P', plot_list, 'OH',title=)

# In[WITHOUT_ATOMS]

Cons=[100]#np.linspace(400,800,1)# Use the loop to look for the best value of K, with the rest of parameters fixed. 
for CO in Cons: 
    print(CO)
    S_I = np.logspace(0, 4, 50)
    kwargs = {'tau_SF': 1.5, 'wind': 2.}
    tau_0 = models.model_run(S_I, model_type='cte', K=CO,  tau_I=1e-3, **kwargs)
    tau_2 = models.model_run(S_I, model_type='cte', K=CO, tau_I=2., **kwargs)
    tau_4 = models.model_run(S_I, model_type='cte', K=CO, tau_I=4., **kwargs)
    tau_inf = models.model_run(S_I, model_type='cte', K=CO, tau_I=1e3, **kwargs)
    
    model_list = [tau_0, tau_2, tau_4, tau_inf] #[tau_0, 
    compute_derived(*model_list)
    
    CALIFA['marker'] = 'o'
    CALIFA['edge'] = 'face'
    CALIFA['size'] = 5
    CALIFA['alpha'] = 0.2
    
    THINGS['marker'] = 's'
    THINGS['edge'] = 'face'
    THINGS['size'] = 5
    THINGS['alpha'] = 1
    
    MW['marker'] = '*'
    MW['edge'] = 'k'
    MW['size'] = 20
    MW['alpha'] = 1
    
    tau_0['color'] = 'red'
    tau_2['color'] = 'red'
    tau_4['color'] = 'blue'
    tau_inf['color'] = 'blue'
    tau_0['style'] = '-'
    tau_2['style'] = '--'
    tau_4['style'] = '-.'
    tau_inf['style'] = ':'
    
    # Parameters for each quantity
    plot_list = ['SFmol', 'Rmol', 'SFE', 'gas', 'OH', 'SSFR','sfr']#'SFmol', 'Rmol', 'SFE', 'gas', 'OH', 'SSFR','sfr'
    column_plot('stars', plot_list, 'OH',title='_cte_')
    #column_plot('P', plot_list, 'OH',title=)
    
    tau_0 = models.model_run(S_I, K=50,  tau_I=1e-3,model_type='variable',eta_dis=10, **kwargs)
    tau_2 = models.model_run(S_I,  K=50, tau_I=2.,model_type='variable',eta_dis=10,  **kwargs)
    tau_4 = models.model_run(S_I,  K=50, tau_I=4.,model_type='variable',eta_dis=10, **kwargs)
    tau_inf = models.model_run(S_I, K=50, tau_I=1e3,model_type='variable',eta_dis=10, **kwargs)
    
    
    model_list = [tau_0, tau_2, tau_4, tau_inf] 
    compute_derived(*model_list)
        
    CALIFA['marker'] = 'o'
    CALIFA['edge'] = 'face'
    CALIFA['size'] = 5
    CALIFA['alpha'] = 0.2
    
    THINGS['marker'] = 's'
    THINGS['edge'] = 'face'
    THINGS['size'] = 5
    THINGS['alpha'] = 1
    
    MW['marker'] = '*'
    MW['edge'] = 'k'
    MW['size'] = 20
    MW['alpha'] = 1
    
    tau_0['color'] = 'red'
    tau_2['color'] = 'red'
    tau_4['color'] = 'blue'
    tau_inf['color'] = 'blue'
    tau_0['style'] = '-'
    tau_2['style'] = '--'
    tau_4['style'] = '-.'
    tau_inf['style'] = ':'
    
    # Parameters for each quantity
    plot_list = ['SFmol', 'Rmol', 'SFE', 'gas', 'OH', 'SSFR','sfr']
    column_plot('stars', plot_list, 'OH',title='_variable_')
    #column_plot('P', plot_list, 'OH',title=)

# <codecell> tests

#column_plot('stars', ['gas'], 'OH')

# <codecell> Bye
# -----------------------------------------------------------------------------
#                                                           ... Paranoy@ Rulz!
# -----------------------------------------------------------------------------