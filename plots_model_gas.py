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
CALIFA['velocity_HI'] = np.empty_like(y)*np.NaN

# THINGS (Leroy et al. 2008) + OH (Kudritzki et al. 2015) + Velocity dispersion (Tamburro et al. 2009)

Z0_str, dZdR_str, R_L08_str, HI_L08_str, H2_L08_str, Stars_L08_str, SFR_L08_str, v_HI_str = \
    np.genfromtxt(os.path.join('./input/L+08_T+09_K+15.txt'),#base_dir, 'input', 'L+08_K+15.txt'),
        usecols=(1, 2, 3, 5, 7, 9, 11, 13), unpack=True)                  
        #usecols=(0, 1, 2, 4, 6, 8, 10, 13), unpack=True)
Z0, dZdR, R_L08, HI_L08, H2_L08, Stars_L08, SFR_L08, v_HI = \
Z0_str.astype(float), dZdR_str.astype(float), R_L08_str.astype(float),\
HI_L08_str.astype(float), H2_L08_str.astype(float), Stars_L08_str.astype(float),\
SFR_L08_str.astype(float), v_HI_str.astype(float)

# r,sfr =  np.loadtxt('./input/Tamburro+09/NGC4214_sfr.txt',#base_dir, 'input', 'L+08_K+15.txt'),
#    usecols=(0, 1), unpack=True)
    
THINGS = {}
THINGS['stars'] = Stars_L08
THINGS['HI'] = HI_L08
THINGS['H2'] = H2_L08
THINGS['gas'] = HI_L08 + H2_L08
THINGS['SFR'] = 0.1*SFR_L08  # Msun/pc^2/Gyr                                
THINGS['OH'] = Z0 - dZdR*R_L08
v_HI[v_HI == 0.] = np.NaN
THINGS['velocity_HI'] = v_HI # km/s

# Milky Way (MollÃ¡ et al. 2015)

MW = {}
MW['stars'] = 10**np.array([np.NaN, np.NaN, np.NaN, 2.43, 2.50, 2.40, 2.25, 2.09, 1.95, 1.79, 1.69, 1.51, 1.38, 1.25, 1.09, 0.94, 0.80, np.NaN, np.NaN, np.NaN, np.NaN])
MW['HI'] = np.array([9.41, 3.97, 2.37, 2.39, 3.86, 5.06, 5.04, 5.44, 5.69, 7.69, 6.52, 6.16, 5.63, 4.83, 3.65, 2.96, 2.42, 2.15, 1.61, 1.18, 1.1])
MW['H2'] = np.array([0.30, 3.82, 5.18, 3.48, 5.69, 8.28, 8.47, 4.59, 3.15, 2.44, 1.96, 1.24, 0.99, 0.57, 0.82, 1.09, .20, .13, .08, .03, np.NaN])
MW['gas'] = MW['HI'] + MW['H2']
MW['SFR'] = 10**np.array([-.37, .603, .706, .983, 1.163, 1.185, 1.181, .963, .723, .594,  .51, .403, .006, .183, -.26, -.132, -.52, -.68, -.89, -1.37, -1.37])
MW['OH'] = np.array([9.02, 8.86, 8.74, 8.62, 8.82, 8.83, 8.77, 8.69, 8.56, 8.60, 8.45, 8.41, 8.44, 8.44, 8.42, 8.14, 8.14, 8.19, 7.96, np.NaN, np.NaN])
MW['velocity_HI'] = np.empty_like(MW['OH'])*np.NaN

# <codecell> Derived observational quantities

def compute_derived(*args):
    '''
    Function to derive:
        P : Pressure
        SFE : Star Formation Efficiecy
        SSFR : Specific Star Formation Rate 
        Rmol : Ratio of molecular to atomic hydrogen
        SFmol : Star Formation rate per molecular density
        s : Number of stars per gas density 
        sfr : Star Formation Rate
        '''
    for dataset in args:
        dataset['P'] = np.pi / 2 * models.G_pc3_Msun_1_Gyr_2 * dataset['gas'] * (dataset['gas'] + dataset['stars']) # Pa
        dataset['SFE'] = dataset['SFR']/dataset['gas']
        dataset['SSFR'] = dataset['SFR']/dataset['stars']
        dataset['s'] = dataset['stars']/dataset['gas']
        dataset['total'] = dataset['stars']+dataset['gas']
        dataset['sfr'] = dataset['SFR']
        dataset['cs'] = 591*(dataset['total']/dataset['gas'])**0.25
data_list = [CALIFA, THINGS, MW]

# Conversion of observations to parameters to plot. 
compute_derived(*data_list)


# <codecell> Models & plotting functions                                               

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

# We define the setting of the observables to plot: title and limits. 
pars = {}
pars['stars'] = plot_pars('$\\Sigma_*$ [M$_\\odot$ pc$^{-2}$]', .3, 3e4)
pars['gas'] = plot_pars('$\\Sigma_{gas}$ [M$_\\odot$ pc$^{-2}$]', 3e-1, 3e3)
pars['total'] = plot_pars('$\\Sigma_{total}$ [M$_\\odot$ pc$^{-2}$]', 3, 3e4)
pars['SSFR'] = plot_pars('SSFR [Gyr$^{-1}$]', 3e-5, 3)
pars['SFE'] = plot_pars('SFE [Gyr$^{-1}$]', 3e-3, 3e1)
pars['OH'] = plot_pars('12 + log(O/H)', 7.9, 9.1, 'linear', 8.25, 8.55)
pars['P'] = plot_pars(
       '$\\Sigma_{gas} (\\Sigma_{gas}+\\Sigma_*)$ [M$_\\odot$ pc$^{-2}$]',
       3e4, 3e10)
pars['Rmol'] = plot_pars('$\\Sigma_{\\rm H_2}$ / $\\Sigma_{\\rm HI}$', 3e-2, 3e2)
pars['SFmol'] = plot_pars('$\Sigma_{\\psi}/\\Sigma_{\\rm H_2}$', 3e-1, 3e1)
pars['sfr'] = plot_pars('SFR[Gyr$^{-1}$]', 3e-2, 3e2)
pars['cs'] = plot_pars('c$_s$[km/s]',500,3000)
pars['velocity_HI'] = plot_pars('\sigma$_{HI}$[km/s]',6,6e1)
# Function definitions

def plot_data(x, y, color, ax):
    """ 
    Plot all datasets. For each, it plots both the observations and the models
    """
    cbar = pars[color]
    for dataset in data_list: # For all the observations, 
        xx = dataset[x]
        yy = dataset[y]
        c = dataset[color]
        m = dataset['marker']
        e = dataset['edge']
        s = dataset['size']
        a = dataset['alpha']   
        sc = ax.scatter(xx, yy, s=s, c=c, marker=m,   # PLOT THE DATA
                        edgecolors=e, linewidths=.2,
                        cmap=cbar.cmap, norm=cbar.norm, alpha=a)
        bad_color = np.where(np.isnan(c))
        ax.plot(xx[bad_color], yy[bad_color], color='gray', linestyle='None',
                marker=m, ms=s, alpha=a*.1)
    for model in model_list: # For the models, PLOT THE MODELS 
        ax.plot(model[x], model[y],
                marker='+', color=model['color'], linestyle=model['style'])

    return sc

def column_plot(x, plots, color, save=True, title=None):
    """ Plot a set of quantities as a function of x, colored by color """
    fs=15 # fontsize
    # Set dimensions

    n_plots = len(plots) # number of plots
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
    for i, y in enumerate(plots):              # Loop over the different plots. 
        ax = fig.add_axes([left, bottom+i*panel_height,
                           panel_width, panel_height])   # add a plot
        ax.set_xlim(xmin, xmax)
        ax.set_xscale(xscale)
        if i == 0:
            ax.set_xlabel(pars[x].title,fontsize=fs)
        ax.set_ylim(pars[y].xmin, pars[y].xmax)
        ax.set_yscale(pars[y].scale)
        ax.set_ylabel(pars[y].title,fontsize=fs)
        ax.tick_params(axis='both', which='both', direction='in',
                       top=True, right=True, labelbottom=(i == 0),labelsize=fs,)
        if pars[y].scale=='log':
            print(pars[y])
            ymin = pars[y].xmin
            ymax = pars[y].xmax
            steps = int(np.log10(ymax))-int(np.log10(ymin))+1
            if steps>4:
                steps = 3
            ax.set_yticks(np.logspace(int(np.log10(ymin)),int(np.log10(ymax)),steps))
        sc = plot_data(x, y, color, ax) #Plotting data and models (see function above)
        
    # Color bar
    ax_cbar = fig.add_axes([left, bottom+n_plots*panel_height+bottom,
                            panel_width, cbar_height])
    cbar = pars[color]
    bar = plt.colorbar(sc, cax=ax_cbar, cmap=cbar.cmap, norm=cbar.norm, orientation='horizontal')
    bar.solids.set_alpha(1)
    bar.ax.tick_params(labelsize=fs)
    ax_cbar.set_title(cbar.title,fontsize=fs)
    if save==True and title==None:
        #fig.savefig(os.path.join('./paper/figs/',x+'Ysfr.pdf'))
        fig.savefig(os.path.join('./paper/figs/',x+'Ysfr.jpeg'))

    elif save==True and title!=None:
        #fig.savefig(os.path.join('./paper/figs/',title+x+'.pdf'))     
        fig.savefig(os.path.join('./paper/figs/',title+x+'.jpeg'))       
   
# <codecell> Marker definition:
#plt.close('all')

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

S_I = np.logspace(0, 4, 10)
kwargs = {'tau_SF' : 0.1}
tau_0 = models.model_run(S_I, infall_time_Gyr=1e-3, **kwargs)
tau_2 = models.model_run(S_I, infall_time_Gyr=2., **kwargs)
tau_4 = models.model_run(S_I, infall_time_Gyr=4., **kwargs)
tau_inf = models.model_run(S_I, infall_time_Gyr=1e3, **kwargs)

model_list = [tau_0, tau_2, tau_4, tau_inf]
compute_derived(*model_list)

tau_0['color'], tau_0['style'] = 'red','-'
tau_2['color'], tau_2['style'] = 'red','--'
tau_4['color'], tau_4['style'] = 'blue', '-.'
tau_inf['color'], tau_inf['style'] = 'blue', ':'

plot_list = ['SFE', 'gas', 'SSFR','sfr','P','cs','velocity_HI']
column_plot('stars', plot_list, 'OH',title='variable_vs_stars')
#plt.title('FF')
column_plot('P', plot_list, 'OH',title='variable_vs_P')
#plt.title('FF')

'''#%%
tau_0 = models.model_run(S_I, infall_time_Gyr=1e-3, **kwargs)
tau_2 = models.model_run(S_I, infall_time_Gyr=2., **kwargs)
tau_4 = models.model_run(S_I, infall_time_Gyr=4., **kwargs)
tau_inf = models.model_run(S_I, infall_time_Gyr=1e3, **kwargs)

fig,[ax1,ax2] = plt.subplots(2)
sc = ax1.scatter(THINGS['stars'],THINGS['velocity_HI'],c=THINGS['OH'],cmap='rainbow')
ax1.grid()
ax1.set_xlabel(r'$\Sigma_{*}$'),ax1.set_ylabel(r'$\sigma_{HI}$[km/s]')
ax1.set_xscale('log'),ax1.set_yscale('log')
ax1.set_xlim(3,3e4),ax1.set_ylim(ymax=60)

Px = np.pi / 2 * models.G_pc3_Msun_1_Gyr_2 * (THINGS['HI'] + THINGS['H2']) * (THINGS['HI'] + THINGS['H2'] + THINGS['stars']) # Pa

ax2.scatter(Px,THINGS['velocity_HI'],c=THINGS['OH'],cmap='rainbow')
ax2.grid()
ax2.set_xlabel(r'$P$'),ax2.set_ylabel(r'$\sigma_{HI}$[km/s]')
ax2.set_xscale('log'),ax2.set_yscale('log')
fig.colorbar(sc,label='12+log(O/H)',location='top',ax=ax1)
ax2.set_xlim(3e4,3e10),ax2.set_ylim(ymax=60)

plt.show()
ax1.plot(tau_0['stars'],tau_0['cs']/80,'red',ls='-')
ax1.plot(tau_2['stars'],tau_2['cs']/80,'red',ls='--')
ax1.plot(tau_4['stars'],tau_4['cs']/80,'blue',ls='-.')
ax1.plot(tau_inf['stars'],tau_inf['cs']/80,'blue',ls=':')


ax2.plot(tau_0['P'],tau_0['cs']/80,'red',ls='-')
ax2.plot(tau_2['P'],tau_2['cs']/80,'red',ls='--')
ax2.plot(tau_4['P'],tau_4['cs']/80,'blue',ls='-.')
ax2.plot(tau_inf['P'],tau_inf['cs']/80,'blue',ls=':')
'''
#%%

Px = np.pi / 2 * models.G_pc3_Msun_1_Gyr_2 * (THINGS['HI'] + THINGS['H2']) * (THINGS['HI'] + THINGS['H2'] + THINGS['stars']) # Pa

fig,[ax1,ax2,ax3] = plt.subplots(3,sharex=True,gridspec_kw={"hspace":0})
Sgas_T09 = THINGS['HI'] + THINGS['H2']
Stot_T09 = Sgas_T09 + THINGS['stars']
ax1.grid(), ax2.grid(), ax3.grid()
ax1.set_xscale('log'), ax1.set_yscale('log')
ax2.set_xscale('log'), ax2.set_yscale('log')
ax3.set_xscale('log'), ax3.set_yscale('log')
fig.supxlabel(r'$\rho$')
ax1.set_ylabel(r'$\tau_{SF}$'), ax2.set_ylabel(r'$t_{SF}$ [Gyr]'), ax3.set_ylabel(r'$t_{ff}$ [Gyr]')

tau_SF = Sgas_T09*Stot_T09*models.G_pc3_Msun_1_Gyr_2/(THINGS['SFR']*THINGS['velocity_HI'])
rho = Px/(THINGS['velocity_HI']**2)*0.175 # proton mass/cm**3
t_SF = Sgas_T09/THINGS['SFR']
t_ff = THINGS['velocity_HI']/np.sqrt(models.G_pc3_Msun_1_Gyr_2*rho)*0.22

sc = ax1.scatter(rho,tau_SF,c=THINGS['OH'],cmap='rainbow',vmin=8.25,vmax=9.)
ax2.scatter(rho,t_SF,c=THINGS['OH'],cmap='rainbow',vmin=8.25,vmax=9.)
ax3.scatter(rho,t_ff,c=THINGS['OH'],cmap='rainbow',vmin=8.25,vmax=9.)

fig.colorbar(sc,label='12+log(O/H)',location='top',ax=ax1)




# <codecell> 

#plot_list = ['SFE']
#column_plot('total', plot_list, 'OH',title='0_variable_vs_stars')
#plt.title('FF')



# <codecell> Bye
# -----------------------------------------------------------------------------
#                                                           ... Paranoy@ Rulz!
# -----------------------------------------------------------------------------