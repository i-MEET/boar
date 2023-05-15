######################################################################
#################### Plot function SIMsalabim ########################
######################################################################
# by Vincent M. Le Corre
# Package import
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from scipy import stats,optimize,constants
import warnings
import sys
# Don't show warnings
warnings.filterwarnings("ignore")
## Physics constants
q = constants.value(u'elementary charge')
eps_0 = constants.value(u'electric constant')
kb = constants.value(u'Boltzmann constant in eV/K')

def make_df_JV(path_JV_file):
    """Output panda dataframe containing JV_file

    Parameters
    ----------
    path_JV_file : str
        path to file containing the JV_file output from SIMsalabim
    
    Returns
    -------
    DataFrame
        panda dataframe containing JV_file
    """    
    #names = ['Vext','Vint','Jext','Jint','P','recLan','recSRH','Jbimo','JSRH_bulk','JSRH_LI','JSRH_RI','Jph','Jn_l','Jp_l','Jn_r','Jp_r']
    df = pd.read_csv(path_JV_file,delim_whitespace=True)

    return df

def make_df_Var(path_Var_file):
    """Output panda dataframe containing Var_file

    Parameters
    ----------
    path_JV_file : str
        path to file containing the Var_file output from SIMsalabim
    
    Returns
    -------
    DataFrame
        panda dataframe containing JV_file
    """   
    #names = ['x','V','n','p','Evac','Ec','Ev','phin','phip','ntrap','ptrap','nid','pid','nion','pion','mun','mup','rec','dp','Gm','Jn','Jp']
    df = pd.read_csv(path_Var_file,delim_whitespace=True)

    return df

def SIMsalabim_nrj_diag(num_fig,data_Var,th_TL_left,th_TL_right,Vext='nan',Background_color=True,no_axis=True,legend=True,pic_save_name='energy_diagram.jpg',save_fig=False):
    """"Make energy diagram plot from Var_file output SIMsalabim

    Parameters
    ----------
    num_fig : int
        figure number where to plot the energy diagram

    data_Var : DataFrame
        Panda Dataframe containing the Var_file output from SIMsalabim (see function "make_df_Var")

    th_TL_left : float
        Thickness of the left transport layer

    th_TL_right : float
        Thickness of the right transport layer

    Vext : float
        float to define the voltage at which the densities will be plotted if Vext='nan' then take Vext as max(Vext), if Vext does not exist we plot the closest voltage, default 'nan'

    Background_color: bool, optional
        Add nice background color to highlight the layer structure, default True

    no_axis : bool, optional
        Chose to show axis or not, default True

    legend : bool, optional
        Display legend or not, by default True

    pic_save_name : str
        name of the file where the figure is saved, default 'energy_diagram.jpg'

    save_fig : bool
        If True, save energy diagram as an image with the  file name defined by "pic_save_name" , default False
    """    
    
    line_thick = 3
    # Color control for the different layers
    color_nTL = '#c7e6a3'
    color_pTL = '#8cb5f0'
    color_pero = '#ba0000'
    color_electrode ='#999999'

    # Check for specific 'Vext'
    if Vext == 'nan':
        Vext = max(data_Var['Vext'])
        print('\n')
        print('In SIMsalabim_nrj_diag function')
        print('Vext was not specified so Vext = {:.2f} V was plotted'.format(Vext))
    
    data_Var = data_Var[abs(data_Var.Vext -Vext) == min(abs(data_Var.Vext -Vext))]
    data_Var = data_Var.reset_index(drop=True)
    if min(abs(data_Var.Vext -Vext)) != 0:
        print('Vext = {:.2f} V was not found so {:.2f} was plotted'.format(Vext,data_Var['Vext'][0]))

    # Convert in nm
    data_Var['x'] = data_Var['x'] * 1e9
    

    # Plotting
    plt.figure(num_fig)
    ax_nrj_diag = plt.axes()
    # ax_nrj_diag.plot('x','Evac',data=data_Var,label = r'E$_{vac}$',linestyle='-',linewidth=2,color = 'k')
    ax_nrj_diag.plot('x','Ec',data=data_Var,label = r'E$_{c}$',linestyle='-', linewidth=line_thick,color = 'k')
    ax_nrj_diag.plot('x','Ev',data=data_Var,label = r'E$_{v}$',linestyle='-', linewidth=line_thick,color = 'k')
    ax_nrj_diag.plot('x','phin',data=data_Var,label = r'E$_{fn}$',linestyle='--',linewidth=line_thick,color = 'k')
    ax_nrj_diag.plot('x','phip',data=data_Var,label = r'E$_{fp}$',linestyle='--',linewidth=line_thick,color = 'k')
    
    if Background_color:
        th_TL_left = th_TL_left* 1e9
        th_TL_right = th_TL_right* 1e9
        TL_left = data_Var[data_Var['x']<th_TL_left]
        TL_right = data_Var[data_Var['x']>max(data_Var['x'])-th_TL_right]
        AL = data_Var[data_Var['x']<max(data_Var['x'])-th_TL_right]
        AL = AL[AL['x']>th_TL_left]
        ax_nrj_diag.fill_between(TL_left['x'],TL_left['Ec'],y2=0,color=color_nTL)
        ax_nrj_diag.fill_between(TL_left['x'],TL_left['Ev'],y2=-8,color=color_nTL)
        ax_nrj_diag.fill_between(TL_right['x'],TL_right['Ec'],y2=0,color=color_pTL)
        ax_nrj_diag.fill_between(TL_right['x'],TL_right['Ev'],y2=-8,color=color_pTL)
        ax_nrj_diag.fill_between(AL['x'],AL['Ec'],y2=0,color=color_pero)
        ax_nrj_diag.fill_between(AL['x'],AL['Ev'],y2=-8,color=color_pero)
        ax_nrj_diag.plot([-10,0],[min(data_Var['phin']),min(data_Var['phin'])],color='k')
        ax_nrj_diag.plot([max(data_Var['x']),max(data_Var['x'])+10],[max(data_Var['phip']),max(data_Var['phip'])],color='k')
        ax_nrj_diag.fill_between([-10,0],[min(data_Var['phin']),min(data_Var['phin'])],y2=-8,color=color_electrode)
        ax_nrj_diag.fill_between([max(data_Var['x']),max(data_Var['x'])+10],[max(data_Var['phip']),max(data_Var['phip'])],y2=-8,color=color_electrode)
    # plt.axhline(y=max(data_Var['phip']), color='k', linestyle='-')

    # Hide axis and spines
    if no_axis:
        ax_nrj_diag.get_xaxis().set_visible(False)
        ax_nrj_diag.get_yaxis().set_visible(False)
        for sides in ['right','left','top','bottom']:
            ax_nrj_diag.spines[sides].set_visible(False)

    # Legend
    if legend:
        plt.legend(loc='center',frameon=False,ncol = 2, bbox_to_anchor=(0.52,1.02),fontsize = 40)
    plt.tight_layout()

    # Save file
    if save_fig:
        plt.savefig(pic_save_name,dpi=300,transparent=True)

def SIMsalabim_JVs_plot(num_fig,data_JV,x='Vext',y=['Jext'],xlimits=[],ylimits=[],plot_type=0,labels='',colors='b',line_type = ['-'],mark='',legend=True,plot_jvexp=False,data_JVexp=pd.DataFrame(),save_fig=False,pic_save_name='JV.jpg'):
    """ Make JV_plot for SIMsalabim  
    
    Parameters
    ----------
    num_fig : int
        number of the fig to plot JV

    data_JV : DataFrame
        Panda DataFrame containing JV_file

    x : str, optional
        xaxis data  (default = 'Vext'), by default 'Vext'

    y : list of str, optional
        yaxis data can be multiple like ['Jext','Jbimo']  (default = ['Jext']), by default ['Jext']

    xlimits : list, optional
        x axis limits if = [] it lets python chose limits, by default []

    ylimits : list, optional
        y axis limits if = [] it lets python chose limits, by default []

    plot_type : int, optional
        type of plot 1 = logx, 2 = logy, 3 = loglog else linlin (default = linlin), by default 0

    labels : str, optional
        label of the JV, by default ''

    colors : str, optional
        color for the JV line, by default 'b'

    line_type : list, optional
        type of line for simulated data plot
        size line_type need to be = size(y), by default ['-']

    mark : str, optional
        type of Marker for the JV, by default ''

    legend : bool, optional
        Display legend or not, by default True

    plot_jvexp : bool, optional
        plot an experimental JV or not, by default False

    data_JVexp : [type], optional
        Panda DataFrame containing experimental JV_file with 'V' the voltage and 'J' the current, by default pd.DataFrame()

    save_fig : bool, optional
        If True, save JV as an image with the  file name defined by "pic_save_name", by default False

    pic_save_name : str, optional
        name of the file where the figure is saved, by default 'JV.jpg'
    """    
        
    if len(y) > len(line_type):
        print('Invalid line_type list, we meed len(y) == len(line_type)')
        print('We will use default line type instead')
        line_type = []
        for counter, value in enumerate(y):
            line_type.append('-')

    plt.figure(num_fig)
    ax_JVs_plot = plt.axes()
    for i,line in zip(y,line_type):
        if plot_type == 1:
            ax_JVs_plot.semilogx(data_JV['Vext'],data_JV[i]/10,color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
            if plot_jvexp:
                ax_JVs_plot.semilogx(data_JVexp['V'],data_JVexp['J']/10,'o',markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
            
        
        elif plot_type == 2:
            ax_JVs_plot.semilogy(data_JV['Vext'],data_JV[i]/10,color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)   
            if plot_jvexp:
                ax_JVs_plot.semilogy(data_JVexp['V'],data_JVexp['J']/10,'o',markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
            
        elif plot_type == 3:
            ax_JVs_plot.loglog(data_JV['Vext'],data_JV[i]/10,color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
            if plot_jvexp:
                ax_JVs_plot.loglog(data_JVexp['V'],data_JVexp['J']/10,'o',markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
            
        else:
            ax_JVs_plot.plot(data_JV['Vext'],data_JV[i]/10,color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
            if plot_jvexp:
                ax_JVs_plot.plot(data_JVexp['V'],data_JVexp['J']/10,'o',markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
        
    
    # legend
    if legend == True:
        plt.legend(loc='best',frameon=False)
    if xlimits != []:
        plt.xlim(xlimits)
    if ylimits != []:
        plt.ylim(ylimits)
    plt.grid(b=True,which='both')
    plt.xlabel('Applied Voltage [V]')
    plt.ylabel('Current Density [mA cm$^{-2}$]')
    plt.tight_layout()
    if save_fig:
        plt.savefig(pic_save_name,dpi=300,transparent=True)

def SIMsalabim_dens_plot(num_fig,data_Var,Vext=['nan'],y=['n','p'],xlimits=[],ylimits=[],x_unit='nm',y_unit='cm^-3',plot_type=0,labels='',colors='b',colorbar_type='None',colorbar_display=False,line_type = ['-','--'],legend=True,save_fig=False,pic_save_name='density.jpg'):
    """Make Var_plot for SIMsalabim

    Parameters
    ----------
    num_fig : int
        number of the fig to plot JV

    data_JV : DataFrame
        Panda DataFrame containing JV_file

    Vext : float
        float to define the voltage at which the densities will be plotted if t='nan' then taketVext as max(t), iftVext does not exist we plot the closest voltage, default ['nan']

    y : list of str, optional
        yaxis data can be multiple like ['n','p'], by default ['n','p']

    xlimits : list, optional
        x axis limits if = [] it lets python chose limits, by default []

    ylimits : list, optional
        y axis limits if = [] it lets python chose limits, by default []

    x_unit : str, optional
        specify unit of the x-axis either ['nm','um','m'], by default 'nm'

    y_unit : str, optional
        specify unit of the y-axis either ['cm^-3','m^-3'], by default 'cm^-3'

    plot_type : int, optional
        type of plot 1 = logx, 2 = logy, 3 = loglog else linlin (default = linlin), by default 0

    labels : str, optional
        label of the line, by default ''

    colors : str, optional
        color for the line, by default 'b'
    
    colorbar_type : str, optional
        define the type of colorbar to use for the plot ['None','log','lin'], by default 'None'
    
    colorbar_display : bool, optional
        chose to display colormap or not, by default False

    line_type : list, optional
        type of line for the plot
        size line_type need to be = size(y), by default ['-']

    legend : bool, optional
        Display legend or not, by default True

    save_fig : bool, optional
        If True, save density plot as an image with the  file name defined by "pic_save_name", by default False

    pic_save_name : str, optional
        name of the file where the figure is saved, by default 'density.jpg'
    """    
    
    if len(y) > len(line_type):
        print('\n')
        print('In SIMsalabim_dens_plot function')
        print('Invalid line_type list, we meed len(y) == len(line_type)')
        print('We will use default line type instead')
        line_type = []
        for counter, value in enumerate(y):
            line_type.append('-')

    if Vext == ['nan']:
        Vext = [max(data_Var['Vext'])]
        print('\n')
        print('In SIMsalabim_dens_plot function')
        print('V was not specified so Vext = {:.2e} V was plotted'.format(Vext[0]))


    # Convert in x-axis
    if x_unit == 'nm':
        data_Var['x'] = data_Var['x'] * 1e9
    elif x_unit == 'um':
        data_Var['x'] = data_Var['x'] * 1e6
    elif x_unit == 'm':
        pass
    else:
        print('\n')
        print('In SIMsalabim_dens_plot function.')
        print('x_unit is wrong so [m] is used. ')
    
    # Convert in y-axis
    if y_unit == 'cm^-3':
        convert_factor = 1e6
    elif y_unit == 'm^-3':
        convert_factor = 1
    else:
        print('\n')
        print('In SIMsalabim_dens_plot function.')
        print('y_unit is wrong so [m^-3] is used. ')
        convert_factor = 1
    
    # Prepare the colorbar is there is any
    if colorbar_type == 'log':
        Vext_bar = data_Var['Vext']
        Vext_bar = np.asarray(Vext_bar.drop_duplicates())
        norm = mp.colors.LogNorm(vmin=np.min(Vext_bar[1]),vmax=np.max(Vext_bar))
        c_m = mp.cm.viridis# choose a colormap
        s_m = mp.cm.ScalarMappable(cmap=c_m, norm=norm)# create a ScalarMappable and initialize a data structure
        s_m.set_array([])
    elif colorbar_type == 'lin':
        Vext_bar = data_Var['Vext']
        Vext_bar = np.asarray(Vext_bar.drop_duplicates())
        norm = mp.colors.Normalize(vmin=np.min(Vext_bar),vmax=np.max(Vext_bar))
        c_m = mp.cm.viridis# choose a colormap
        s_m = mp.cm.ScalarMappable(cmap=c_m, norm=norm) # create a ScalarMappable and initialize a data structure
        s_m.set_array([])
    elif colorbar_type == 'None':
        pass
    else:
        print('Wrong colorbar_type input')

       
    for V in Vext:
        data_Var_dum = data_Var[abs(data_Var.Vext -V) == min(abs(data_Var.Vext -V))]
        data_Var_dum = data_Var_dum.reset_index(drop=True)
        if min(abs(data_Var.Vext -V)) != 0:
            print('Vext = {:.2e} V was not found so {:.2e} V was plotted'.format(V,data_Var_dum['Vext'][0]))
        
        plt.figure(num_fig)
        ax_Vars_plot = plt.axes()
        if (colorbar_type == 'log' or colorbar_type == 'lin') and colorbar_display:
            colorline = s_m.to_rgba(V)
        else:
            colorline = colors

        put_label = True
        for i,line in zip(y,line_type):
            if put_label:
                labels = labels
                put_label = False
            else:
                labels = ''

            if plot_type == 1:
                ax_Vars_plot.semilogx(data_Var_dum['x'],data_Var_dum[i]/convert_factor,color=colorline,label=labels,linestyle=line)
                        
            
            elif plot_type == 2:
                ax_Vars_plot.semilogy(data_Var_dum['x'],data_Var_dum[i]/convert_factor,color=colorline,label=labels,linestyle=line)   
                
            elif plot_type == 3:
                ax_Vars_plot.loglog(data_Var_dum['x'],data_Var_dum[i]/convert_factor,color=colorline,label=labels,linestyle=line)
                
            else:
                ax_Vars_plot.plot(data_Var_dum['x'],data_Var_dum[i]/convert_factor,color=colorline,label=labels,linestyle=line)
            
    
    # legend
    if legend == True:
        plt.legend(loc='best',frameon=False,fontsize = 30)
    
    # Add colorbar if needed
    if (colorbar_type == 'log' or colorbar_type == 'lin') and colorbar_display:
        cbar = plt.colorbar(s_m)
        cbar.set_label('Vext [V]')

    # Set axis limits
    if xlimits != []:
        plt.xlim(xlimits)
    if ylimits != []:
        plt.ylim(ylimits)
    plt.grid(b=True,which='both')
    
    # Label x-axis
    if x_unit == 'nm':
        plt.xlabel('x [nm]')
    elif x_unit == 'um':
        plt.xlabel('x [$\mu$m]')
    elif x_unit == 'm':
        plt.xlabel('x [m]')
    else:
        plt.xlabel('x [nm]')
    
    # Label y-axis
    if y_unit == 'cm^-3':
        plt.ylabel('Density [cm$^{-3}$]')
    elif y_unit == 'm^-3':
        plt.ylabel('Density [m$^{-3}$]')
    else:
        plt.ylabel('Density [m$^{-3}$]')
    
    plt.ylabel('Density [cm$^{-3}$]')
    plt.tight_layout()
    if save_fig:
        plt.savefig(pic_save_name,dpi=300,transparent=True)