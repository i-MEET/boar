#####################################################################
###################### Plot functions ZimT ###########################
#####################################################################
# Author: Vincent M. Le Corre
# Github: https://github.com/VMLC-PV

# Import libraries
import math,sys
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from scipy import constants

## Physics constants
q = constants.value(u'elementary charge')
eps_0 = constants.value(u'electric constant')
kb = constants.value(u'Boltzmann constant in eV/K')


def zimt_tj_plot(num_fig,data_tj,x='t',y=['Jext'],xlimits=[],ylimits=[],plot_type=0,labels='',colors='b',line_type = ['-'],mark='',legend=True,save_yes=False,pic_save_name='transient.jpg'):
    """ Make tj_file transient plot for ZimT  
    Default time on the x axis in $\mu$s
    
    Parameters
    ----------
    num_fig : int
        number of the fig to plot tj

    data_tj : DataFrame
        Panda DataFrame containing tj_file

    x : str, optional
        xaxis data, by default 't'

    y : list of str, optional
        yaxis data can be multiple like ['Jext','Jncat'], by default ['Jext']

    xlimits : list, optional
        x axis limits if = [] it lets python chose limits, by default []

    ylimits : list, optional
        y axis limits if = [] it lets python chose limits, by default []

    plot_type : int, optional
        type of plot 1 = logx, 2 = logy, 3 = loglog else linlin (default = linlin), by default 0

    labels : str, optional
        label of the tj, by default ''

    colors : str, optional
        color for the JV line, by default 'b'

    line_type : list, optional
        type of line used for the plot
        size line_type need to be = size(y), by default ['-']

    mark : str, optional
        type of Marker used for the plot, by default ''

    legend : bool, optional
        Display legend or not, by default True

    pic_save_name : str, optional
        name of the file where the figure is saved, by default 'transient.jpg'
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
            ax_JVs_plot.semilogx(data_tj[x],data_tj[i]/10,color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
            
        elif plot_type == 2:
            ax_JVs_plot.semilogy(data_tj[x],data_tj[i]/10,color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
            
        elif plot_type == 3:
            ax_JVs_plot.loglog(data_tj[x],data_tj[i]/10,color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
            
        else:
            ax_JVs_plot.plot(data_tj[x],data_tj[i]/10,color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)

    # legend
    if legend == True:
        plt.legend(loc='best',frameon=False,fontsize = 30)
    
    if xlimits != []:
        plt.xlim(xlimits)
    if ylimits != []:
        plt.ylim(ylimits)

    plt.grid(visible=True,which='both')
    plt.xlabel('Time [s]')
    plt.ylabel('Current Density [mA cm$^{-2}$]')
    plt.tight_layout()
    if save_yes:
        plt.savefig(pic_save_name,dpi=300,transparent=True)

def zimt_tj_JV_plot(num_fig,data_tj,x='Vext',y=['Jext'],xlimits=[],ylimits=[],plot_type=0,labels='',colors='b',line_type = ['-'],mark='',legend=True,save_yes='False',pic_save_name='transient_JV.jpg'):
    """ Make tj_file transient current-voltage curve plot for ZimT  
    Default Voltage on the x axis 
    
    Parameters
    ----------
    num_fig : int
        number of the fig to plot tj

    data_tj : DataFrame
        Panda DataFrame containing tj_file

    x : str, optional
        xaxis data, by default 'Vext'

    y : list of str, optional
        yaxis data can be multiple like ['Jext','Jncat'], by default ['Jext']

    xlimits : list, optional
        x axis limits if = [] it lets python chose limits, by default []

    ylimits : list, optional
        y axis limits if = [] it lets python chose limits, by default []

    plot_type : int, optional
        type of plot 1 = logx, 2 = logy, 3 = loglog else linlin (default = linlin), by default 0

    labels : str, optional
        label of the tj, by default ''

    colors : str, optional
        color for the JV line, by default 'b'

    line_type : list, optional
        type of line used for the plot
        size line_type need to be = size(y), by default ['-']

    mark : str, optional
        type of Marker used for the plot, by default ''

    legend : bool, optional
        Display legend or not, by default True

    save_yes : bool, optional
        If True, save JV as an image with the  file name defined by "pic_save_name", by default False

    pic_save_name : str, optional
        name of the file where the figure is saved, by default 'transient_JV.jpg'
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
            ax_JVs_plot.semilogx(data_tj[x],data_tj[i]/10,color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
            
        elif plot_type == 2:
            ax_JVs_plot.semilogy(data_tj[x],data_tj[i]/10,color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
            
        elif plot_type == 3:
            ax_JVs_plot.loglog(data_tj[x],data_tj[i]/10,color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
            
        else:
            ax_JVs_plot.plot(data_tj[x],data_tj[i]/10,color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)

    # legend
    if legend == True:
        plt.legend(loc='best',frameon=False,fontsize = 30)
    if xlimits != []:
        plt.xlim(xlimits)
    if ylimits != []:
        plt.ylim(ylimits)
    plt.grid(visible=True,which='both')
    plt.xlabel('Voltage [V]')
    plt.ylabel('Current Density [mA cm$^{-2}$]')
    plt.tight_layout()
    if save_yes:
        plt.savefig(pic_save_name,dpi=300,transparent=True)


def zimt_Voltage_transient_plot(num_fig,data_tj,x='t',y=['Vext'],xlimits=[],ylimits=[],plot_type=0,labels='',colors='b',line_type = ['-'],mark='',legend=True,save_yes=False,pic_save_name='transient_volt.jpg'):
    """ Make tj_file transientvoltage curve plot for ZimT  
    Default time on the x axis in $\mu$s
    
    Parameters
    ----------
    num_fig : int
        number of the fig to plot tj

    data_tj : DataFrame
        Panda DataFrame containing tj_file

    x : str, optional
        xaxis data, by default 't'

    y : list of str, optional
        yaxis data can be multiple like ['Vext','Va'], by default ['Vext']

    xlimits : list, optional
        x axis limits if = [] it lets python chose limits, by default []

    ylimits : list, optional
        y axis limits if = [] it lets python chose limits, by default []

    plot_type : int, optional
        type of plot 1 = logx, 2 = logy, 3 = loglog else linlin (default = linlin), by default 0

    labels : str, optional
        label of the tj, by default ''

    colors : str, optional
        color for the JV line, by default 'b'

    line_type : list, optional
        type of line used for the plot
        size line_type need to be = size(y), by default ['-']

    mark : str, optional
        type of Marker used for the plot, by default ''

    legend : bool, optional
        Display legend or not, by default True

    save_yes : bool, optional
        If True, save JV as an image with the  file name defined by "pic_save_name", by default False

    pic_save_name : str, optional
        name of the file where the figure is saved, by default 'transient_volt.jpg'
    """  

    if len(y) != len(line_type):
        print('Invalid line_type list, we meed len(y) == len(line_type)')
        print('We will use default line type instead')
        line_type = []
        for counter, value in enumerate(y):
            line_type.append('-')

    plt.figure(num_fig)
    ax_JVs_plot = plt.axes()
    for i,line in zip(y,line_type):
        if plot_type == 1:
            ax_JVs_plot.semilogx(data_tj[x],data_tj[i],color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
            
        elif plot_type == 2:
            ax_JVs_plot.semilogy(data_tj[x],data_tj[i],color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
            
        elif plot_type == 3:
            ax_JVs_plot.loglog(data_tj[x],data_tj[i],color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
            
        else:
            ax_JVs_plot.plot(data_tj[x],data_tj[i],color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)

    # legend
    if legend == True:
        plt.legend(loc='best',frameon=False,fontsize = 30)
    if xlimits != []:
        plt.xlim(xlimits)
    if ylimits != []:
        plt.ylim(ylimits)
    plt.grid(visible=True,which='both')
    plt.xlabel('Time [s]')
    plt.ylabel('Volatge [V]')
    plt.tight_layout()
    if save_yes:
        plt.savefig(pic_save_name,dpi=300,transparent=True)

def zimt_dens_plot(num_fig,data_Var,time=['nan'],y=['n','p'],xlimits=[],ylimits=[],x_unit='nm',y_unit='cm^-3',plot_type=0,labels='',colors='b',colorbar_type='None',colorbar_display=False,line_type = ['-','--'],legend=True,save_yes=False,pic_save_name='density.jpg'):
    """Make Var_plot for ZimT

    Parameters
    ----------
    num_fig : int
        number of the fig to plot JV

    data_JV : DataFrame
        Panda DataFrame containing JV_file

    time : float
        float to define the voltage at which the densities will be plotted if t='nan' then takettime as max(t), ifttime does not exist we plot the closest voltage, default ['nan']

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

    save_yes : bool, optional
        If True, save density plot as an image with the  file name defined by "pic_save_name", by default False

    pic_save_name : str, optional
        name of the file where the figure is saved, by default 'density.jpg'
    """    
    
    if len(y) > len(line_type):
        print('\n')
        print('In zimt_dens_plot function')
        print('Invalid line_type list, we meed len(y) == len(line_type)')
        print('We will use default line type instead')
        line_type = []
        for counter, value in enumerate(y):
            line_type.append('-')

    if time == ['nan']:
        time = [max(data_Var['time'])]
        print('\n')
        print('In zimt_dens_plot function')
        print('t was not specified so time = {:.2e} s was plotted'.format(time[0]))


    # Convert in x-axis
    if x_unit == 'nm':
        data_Var['x'] = data_Var['x'] * 1e9
    elif x_unit == 'um':
        data_Var['x'] = data_Var['x'] * 1e6
    elif x_unit == 'm':
        pass
    else:
        print('\n')
        print('In zimt_dens_plot function.')
        print('x_unit is wrong so [m] is used. ')
    
    # Convert in y-axis
    if y_unit == 'cm^-3':
        convert_factor = 1e6
    elif y_unit == 'm^-3':
        convert_factor = 1
    else:
        print('\n')
        print('In zimt_dens_plot function.')
        print('y_unit is wrong so [m^-3] is used. ')
        convert_factor = 1
    
    # Prepare the colorbar is there is any
    if colorbar_type == 'log':
        time_bar = data_Var['time']
        time_bar = np.asarray(time_bar.drop_duplicates())
        norm = mp.colors.LogNorm(vmin=np.min(time_bar[1]),vmax=np.max(time_bar))
        c_m = mp.cm.viridis# choose a colormap
        s_m = mp.cm.ScalarMappable(cmap=c_m, norm=norm)# create a ScalarMappable and initialize a data structure
        s_m.set_array([])
    elif colorbar_type == 'lin':
        time_bar = data_Var['time']
        time_bar = np.asarray(time_bar.drop_duplicates())
        norm = mp.colors.Normalize(vmin=np.min(time_bar),vmax=np.max(time_bar))
        c_m = mp.cm.viridis# choose a colormap
        s_m = mp.cm.ScalarMappable(cmap=c_m, norm=norm) # create a ScalarMappable and initialize a data structure
        s_m.set_array([])
    elif colorbar_type == 'None':
        pass
    else:
        print('Wrong colorbar_type input')

       
    for t in time:
        data_Var_dum = data_Var[abs(data_Var.time -t) == min(abs(data_Var.time -t))]
        data_Var_dum = data_Var_dum.reset_index(drop=True)
        if min(abs(data_Var.time -t)) != 0:
            print('time = {:.2e} s was not found so {:.2e} was plotted'.format(t,data_Var_dum['time'][0]))
        
        plt.figure(num_fig)
        ax_Vars_plot = plt.axes()
        if (colorbar_type == 'log' or colorbar_type == 'lin') and colorbar_display:
            colorline = s_m.to_rgba(t)
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
        cbar.set_label('Time [s]')

    # Set axis limits
    if xlimits != []:
        plt.xlim(xlimits)
    if ylimits != []:
        plt.ylim(ylimits)
    plt.grid(visible=True,which='both')
    
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
    if save_yes:
        plt.savefig(pic_save_name,dpi=300,transparent=True)


