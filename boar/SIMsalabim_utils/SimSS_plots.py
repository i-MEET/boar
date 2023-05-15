######################################################################
#################### Plot functions SIMsalabim #######################
######################################################################
# Author: Vincent M. Le Corre
# Github: https://github.com/VMLC-PV

# Import libraries
import itertools,os
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
# Import SIMsalabim_utils
from boar.SIMsalabim_utils.GetInputPar import ReadParameterFile


def PlotJV(JV_files,labels=None,data_type=0,colors= [],num_fig=0,x='Vext',y=['Jext'],xlimits=[],ylimits=[],x_unit='V',y_unit='A/m^2',absx=False,absy=False,plot_type=0,line_type = ['-'],mark='',legend=True,save_fig=False,fig_name='JV.jpg',verbose=True):
    """ Make JV_plot for SIMsalabim 
    Either from a list of JV files or from a dataframe containing JV data 
    
    Parameters
    ----------
    JV_files : list
        List of files containing the JV filenames.
    
    labels : list
        List of labels for the JV_files or string when plotting the data dataframe. 
    
    data_type : int, optional
        Type of data to plot. Either 0 a list of JV filenames or 1 a dataframe with the JV data. The default is 0.
        
    colors : list, optional
        List of colors for the JV_files or string when plotting the data dataframe, by default [].

    num_fig : int
        number of the fig to plot JV, by default 0.

    x : str, optional
        xaxis data  (default = 'Vext'), by default 'Vext'

    y : list of str, optional
        yaxis data can be multiple like ['Jext','Jbimo']  (default = ['Jext']), by default ['Jext']

    xlimits : list, optional
        x axis limits if = [] it lets python chose limits, by default []

    ylimits : list, optional
        y axis limits if = [] it lets python chose limits, by default []
    
    x_unit : str, optional
        unit to plot the x-axis, can be ['mV','V'], by default 'V'

    y_unit : str, optional
        unit to plot the y-axis, can be ['mA/cm^2','A/m^2'], by default 'A/m^2'
    
    absx : bool, optional
        if True, plot the absolute value of x, by default False

    absy : bool, optional
        if True, plot the absolute value of y, by default False

    plot_type : int, optional
        type of plot 1 = logx, 2 = logy, 3 = loglog else linlin (default = linlin), by default 0

    line_type : list, optional
        type of line for simulated data plot
        size line_type need to be = size(y), by default ['-']

    mark : str, optional
        type of Marker for the JV, by default ''

    legend : bool, optional
        Display legend or not, by default True

    save_fig : bool, optional
        If True, save JV as an image with the  file name defined by "fig_name", by default False

    fig_name : str, optional
        name of the file where the figure is saved, by default 'JV.jpg'
    
    verbose : bool, optional
        If True, print some information, by default True
    """    

    if JV_files is None :
        raise ValueError('No JV files or data provided')

    if len(y) > len(line_type):
        if verbose:
            print('Invalid line_type list, we meed len(y) == len(line_type)')
            print('We will use default line type instead')
        line_type = []
        for counter, value in enumerate(y):
            line_type.append('-')

    
    plt.figure(num_fig)
    ax_JVs_plot = plt.gca()
    
    # Convert in x-axis
    if x_unit == 'mV':
        xunit_fact = 1e3
        xaxis_label = 'Applied Voltage [mV]'
    elif x_unit == 'V':
        xunit_fact = 1
        xaxis_label = 'Applied Voltage [V]'
    else:
        xunit_fact = 1
        xaxis_label = 'Applied Voltage [V]'
        print('\n')
        print('In SimssPlotJV function.')
        print('x_unit is wrong so [V] is used. ')
    
    # Convert in y-axis
    if y_unit == 'mA/cm^2':
        yunit_fact = 1e1
        yaxis_label = 'Current Density [mA cm$^{-2}$]'
    elif y_unit == 'A/m^2':
        yunit_fact = 1
        yaxis_label = 'Current Density [A m$^{-2}$]'
    else:
        yunit_fact = 1
        yaxis_label = 'Current Density [A m$^{-2}$]'
        if verbose:
            print('\n')
            print('In SimssPlotJV function.')
            print('y_unit is wrong so [A/m^2] is used. ') 
    
    if colors == [] and JV_files is not None:
        colors = ['b','r','g','c','m','y','k']
    elif colors == [] :
        colors = 'b'

    if data_type == 0:
        for JV,lab,c in zip(JV_files,labels,itertools.cycle(colors)):
            data_JV = pd.read_csv(JV,delim_whitespace=True) # load data
            for i,line in zip(y,line_type):
                if absx:
                    xplot = np.abs(data_JV[x])
                else:
                    xplot = data_JV[x]
                if absy:
                    yplot = np.abs(data_JV[i])
                else:
                    yplot = data_JV[i]
                ax_JVs_plot.plot(xplot/xunit_fact,yplot/yunit_fact,color=c,label=lab,linestyle=line,marker=mark,markeredgecolor=c,markersize=10,markerfacecolor='None',markeredgewidth = 3)  
    elif data_type == 1:
        data_JV = JV_files # load data
        for i,line in zip(y,line_type):
            if absx:
                xplot = np.abs(data_JV[x])
            else:
                xplot = data_JV[x]
            if absy:
                yplot = np.abs(data_JV[i])
            else:
                yplot = data_JV[i]
            ax_JVs_plot.plot(xplot/xunit_fact,yplot/yunit_fact,color=colors,label=labels,linestyle=line,marker=mark,markeredgecolor=colors,markersize=10,markerfacecolor='None',markeredgewidth = 3)
    else:
        raise ValueError('Invalid data_type. It can be 0 or 1')

    # Plot settings
    if plot_type == 1:
        ax_JVs_plot.set_xscale('log')
    elif plot_type == 2:
        ax_JVs_plot.set_yscale('log')
    elif plot_type == 3:
        ax_JVs_plot.set_xscale('log')
        ax_JVs_plot.set_yscale('log')
    else:
        pass

    # legend
    if legend == True:
        plt.legend(loc='best',frameon=False)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)

    # axis limits
    if xlimits != []:
        plt.xlim(xlimits)
    if ylimits != []:
        plt.ylim(ylimits)
    plt.grid(visible=True,which='both')
    plt.tight_layout()

    # Save figure
    if save_fig:
        plt.savefig(fig_name,dpi=300,transparent=True)

def PlotJVPerf(x,scPars_files,y='PCE',Gfrac=[],color='b',xlabel='Time [s]',ylabel='PCE [%]',norm_plot=True,norm_factor=0,legend='',num_fig=5,plot_type=0,xlimits=[],ylimits=[],save_fig=True,fig_name='JV_perf.png',mark='o',line_type = ['-'],verbose=True):
    """Plot the performance of the JV curve from the scPars files

    Parameters
    ----------
    x : list
        x-axis values

    scPars_files : list
        list of filenames containing the scPars

    y : str, optional
        performance name to plot, see scPars output to check available names, by default 'PCE'

    Gfrac : list, optional
        list of Gfrac to correct MPP for light intensity and calculate PCE, need to be the same size as scPars_file, by default []

    color : str, optional
        color of the plot, by default 'b'

    xlabel : str, optional
        x-axis label, by default 'Time [s]'

    ylabel : str, optional
        y-axis label, by default 'PCE [%]'

    norm_plot : bool, optional
        whether of not to normalize the plot, by default True
    
    norm_factor : float, optional
        normalization factor if 0 then normalize using the maximum, by default 0

    legend : str, optional
        legend, by default ''

    num_fig : int, optional
        number of the fig to plot, by default 5

    plot_type : int, optional
        type of plot 1 = logx, 2 = logy, 3 = loglog else linlin (default = linlin), by default 0, by default 0

    xlimits : list, optional
        x axis limits if = [] it lets python chose limits, by default []

    ylimits : list, optional
        y axis limits if = [] it lets python chose limits, by default []

    save_fig : bool, optional
        If True, save density plot as an image with the  file name defined by "fig_name", by default False

    fig_name : str, optional
        name of the file where the figure is saved, by default 'perf.jpg'

    mark : str, optional
        type of marker for the plot, by default 'o'
    
    line_type : list, optional
        type of line for the plot
        size line_type need to be = size(y), by default ['-']
    
    verbose : bool, optional
        If True, print some information, by default True
    """    

    perf_lst = []
    for scpar_name in scPars_files:
        
        perf = pd.read_csv(scpar_name,delim_whitespace=True)
        
        perf_lst.append(list(perf.iloc[0]))
    names = list(perf.columns)
    
    perfs = pd.DataFrame(perf_lst,columns=names)
    if len(Gfrac)>0:
        Gfrac = np.array(Gfrac)
    else:
        Gfrac = 1
    
    if y == 'PCE':
        
        perfs['PCE'] = abs((perfs['Voc']*perfs['FF']*perfs['Jsc'])/(10*Gfrac))
    
    if y == 'Jsc':
        perfs['Jsc'] = abs(perfs['Jsc'])
        
    if norm_plot:
        if norm_factor == 0:
            norm_factor = perfs[y].max()
        else:
            pass
    else:
        norm_factor = 1
    plt.figure(num_fig)
    ax = plt.gca()
    ax.plot(x,perfs[y]/norm_factor,mark,color=color,label=legend,linestyle=line_type[0])


    # Plot controls
    # plot type
    if plot_type == 1:
        ax.set_xscale('log')
    elif plot_type == 2:
        ax.set_yscale('log')
    elif plot_type == 3:
        ax.set_yscale('log')
        ax.set_xscale('log')
    else:
        pass
    #legend
    if legend != '':
        ax.legend(loc='best')
    # axis control
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Set axis limits
    if xlimits != []:
        plt.xlim(xlimits)
    if ylimits != []:
        plt.ylim(ylimits)
    plt.grid(visible=True,which='both')
    plt.tight_layout()


def PlotNrjDiagSimSS(Var_files,labels,path2simu,colors=[],num_fig=0,Vext='nan',x_unit='nm',Background_color=True,show_axis=True,legend=True,fig_name='energy_diagram.jpg',save_fig=False,verbose=True):
    """"Make energy diagram plot from Var_file output SimSS

    Parameters
    ----------
    Var_files : list
        List of files containing the Var filenames.
    
    labels : list
        List of labels for the Var_files.
    
    path2simu : str
        Path to the simulation folder.

    colors : list, optional
        List of colors for the Var_files, by default [].

    num_fig : int
        number of the fig to plot JV, by default 0.

    Vext : float
        float to define the voltage at which the densities will be plotted if Vext='nan' then take Vext as max(Vext), if Vext does not exist we plot the closest voltage, default 'nan'
    
    x_unit : str, optional
        unit to plot the x-axis, can be ['nm','um','m'], by default 'nm'

    Background_color: bool, optional
        Add nice background color to highlight the layer structure, default True

    show_axis : bool, optional
        Choose to show axis or not, default True

    legend : bool, optional
        Display legend or not, by default True

    fig_name : str
        name of the file where the figure is saved, default 'energy_diagram.jpg'

    save_fig : bool
        If True, save energy diagram as an image with the  file name defined by "fig_name" , default False
    
    verbose : bool, optional
        If True, print some information, by default True
    """    
    
    line_thick = 1
    # Color control for the different layers
    color_nTL = '#c7e6a3'
    color_pTL = '#8cb5f0'
    color_pero = '#ba0000'
    color_electrode ='#999999'

    
    # Convert in x-axis
    if x_unit == 'nm':
        xunit_fact = 1e9
        xaxis_label = 'x [nm]'
    elif x_unit == 'um':
        xunit_fact = 1e6
        xaxis_label = 'x [$\mu$m]'
    elif x_unit == 'm':
        xunit_fact = 1
        xaxis_label = 'x [m]'
    else:
        if verbose:
            print('\n')
            print('In PlotNrjDiagSimSS function.')
            print('x_unit is wrong so [m] is used. ')
        xunit_fact = 1
        xaxis_label = 'x [m]'

    # Plotting
    plt.figure(num_fig)
    ax_nrj_diag = plt.gca()

    if colors == []:
        colors = ['k']

    for Var,lab,c in zip(Var_files,labels,itertools.cycle(colors)):
        
        data_Var = pd.read_csv(Var,delim_whitespace=True) # load data

        # Find the voltage at which the diagrams will be plotted
        # Check for specific 'Vext'
        if Vext == 'nan':
            Vext = max(data_Var['Vext'])
            if verbose:
                print('\n')
                print('In PlotNrjDiagSimSS function')
                print('Vext was not specified so Vext = {:.2f} V was plotted'.format(Vext))
        
        data_Var = data_Var[abs(data_Var.Vext -Vext) == min(abs(data_Var.Vext -Vext))]
        data_Var = data_Var.reset_index(drop=True)
        if min(abs(data_Var.Vext -Vext)) != 0:
            if verbose:
                print('Vext = {:.2f} V was not found so {:.2f} was plotted'.format(Vext,data_Var['Vext'][0]))


        data_Var['x'] = data_Var['x']*xunit_fact
        # ax_nrj_diag.plot('x','Evac',data=data_Var,label = r'E$_{vac}$',linestyle='-',linewidth=2,color = 'k')
        ax_nrj_diag.plot('x','Ec',data=data_Var,label = r'E$_{c}$',linestyle='-', linewidth=line_thick,color = c)
        ax_nrj_diag.plot('x','Ev',data=data_Var,label = r'E$_{v}$',linestyle='-', linewidth=line_thick,color = c)
        ax_nrj_diag.plot('x','phin',data=data_Var,label = r'E$_{fn}$',linestyle='--',linewidth=line_thick,color = c)
        ax_nrj_diag.plot('x','phip',data=data_Var,label = r'E$_{fp}$',linestyle='--',linewidth=line_thick,color = c)

    if Background_color:
        # Get the thickness of the transport layers
        ParFileDic = ReadParameterFile(os.path.join(path2simu, 'device_parameters.txt')) # read the parameters from the file
        L_LTL = float(ParFileDic['L_LTL'])*xunit_fact
        L_RTL = float(ParFileDic['L_RTL'])*xunit_fact

        TL_left = data_Var[data_Var['x']<L_LTL]
        TL_right = data_Var[data_Var['x']>max(data_Var['x'])-L_RTL]
        AL = data_Var[data_Var['x']<max(data_Var['x'])-L_RTL]
        AL = AL[AL['x']>L_LTL]
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


    # Hide axis and spines
    if show_axis:
        ax_nrj_diag.get_xaxis().set_visible(True)
        ax_nrj_diag.get_yaxis().set_visible(True)
        ax_nrj_diag.set_xlabel(xaxis_label)
        ax_nrj_diag.set_ylabel(r'Energy [eV]')
    else :
        ax_nrj_diag.get_xaxis().set_visible(False)
        ax_nrj_diag.get_yaxis().set_visible(False)
        for sides in ['right','left','top','bottom']:
            ax_nrj_diag.spines[sides].set_visible(False)
        

    # Legend
    if legend:
        legend_elements = [Line2D([0], [0], color='k', linewidth=line_thick, label='E$_{c}$,E$_{v}$',linestyle='-'),Line2D([0], [0], color='k', linewidth=line_thick, label='E$_{c}$,E$_{v}$',linestyle='--')]
        ax_nrj_diag.legend(handles=legend_elements,loc='upper center',frameon=False,ncol = 2)
    plt.tight_layout()

    # Save file
    if save_fig:
        plt.savefig(fig_name,dpi=300,transparent=True)

def PlotNrjDiagWithTime(Var_files,labels,path2simu,colors=[],num_fig=0,time='nan',x_unit='nm',Background_color=True,show_axis=True,legend=True,fig_name='energy_diagram.jpg',save_fig=False,verbose=True):
    """"Make energy diagram plot from Var_file output SimSS

    Parameters
    ----------
    Var_files : list
        List of files containing the Var filenames.
    
    labels : list
        List of labels for the Var_files.
    
    path2simu : str
        Path to the simulation folder.

    colors : list, optional
        List of colors for the Var_files, by default [].

    num_fig : int
        number of the fig to plot JV, by default 0.

    time : float
        float to define the time at which the densities will be plotted if time='nan' then take time as max(time), if time does not exist we plot the closest time, default 'nan'
    
    x_unit : str, optional
        unit to plot the x-axis, can be ['nm','um','m'], by default 'nm'

    Background_color: bool, optional
        Add nice background color to highlight the layer structure, default True

    show_axis : bool, optional
        Choose to show axis or not, default True

    legend : bool, optional
        Display legend or not, by default True

    fig_name : str
        name of the file where the figure is saved, default 'energy_diagram.jpg'

    save_fig : bool
        If True, save energy diagram as an image with the  file name defined by "fig_name" , default False
    
    verbose : bool, optional
        If True, print some information, by default True
    """    
    
    line_thick = 1
    # Color control for the different layers
    color_nTL = '#c7e6a3'
    color_pTL = '#8cb5f0'
    color_pero = '#ba0000'
    color_electrode ='#999999'

    
    # Convert in x-axis
    if x_unit == 'nm':
        xunit_fact = 1e9
        xaxis_label = 'x [nm]'
    elif x_unit == 'um':
        xunit_fact = 1e6
        xaxis_label = 'x [$\mu$m]'
    elif x_unit == 'm':
        xunit_fact = 1
        xaxis_label = 'x [m]'
    else:
        if verbose:
            print('\n')
            print('In PlotNrjDiagWithTime function.')
            print('x_unit is wrong so [m] is used. ')
        xunit_fact = 1
        xaxis_label = 'x [m]'

    # Plotting
    plt.figure(num_fig)
    ax_nrj_diag = plt.gca()

    if colors == []:
        colors = ['k']

    for Var,lab,c in zip(Var_files,labels,itertools.cycle(colors)):
        
        data_Var = pd.read_csv(Var,delim_whitespace=True) # load data

        # Find the voltage at which the diagrams will be plotted
        # Check for specific 'time'
        if time == 'nan':
            time = max(data_Var['time'])
            if verbose:
                print('\n')
                print('In PlotNrjDiagWithTime function')
                print('time was not specified so time = {:.2f} s was plotted'.format(time))
        
        data_Var = data_Var[abs(data_Var.time -time) == min(abs(data_Var.time -time))]
        data_Var = data_Var.reset_index(drop=True)
        if min(abs(data_Var.time -time)) != 0:
            if verbose:
                print('time = {:.2f} V was not found so {:.2f} was plotted'.format(time,data_Var['time'][0]))


        data_Var['x'] = data_Var['x']*xunit_fact
        # ax_nrj_diag.plot('x','Evac',data=data_Var,label = r'E$_{vac}$',linestyle='-',linewidth=2,color = 'k')
        ax_nrj_diag.plot('x','Ec',data=data_Var,label = r'E$_{c}$',linestyle='-', linewidth=line_thick,color = c)
        ax_nrj_diag.plot('x','Ev',data=data_Var,label = r'E$_{v}$',linestyle='-', linewidth=line_thick,color = c)
        ax_nrj_diag.plot('x','phin',data=data_Var,label = r'E$_{fn}$',linestyle='--',linewidth=line_thick,color = c)
        ax_nrj_diag.plot('x','phip',data=data_Var,label = r'E$_{fp}$',linestyle='--',linewidth=line_thick,color = c)

    if Background_color:
        # Get the thickness of the transport layers
        ParFileDic = ReadParameterFile(os.path.join(path2simu, 'device_parameters.txt')) # read the parameters from the file
        L_LTL = float(ParFileDic['L_LTL'])*xunit_fact
        L_RTL = float(ParFileDic['L_RTL'])*xunit_fact

        TL_left = data_Var[data_Var['x']<L_LTL]
        TL_right = data_Var[data_Var['x']>max(data_Var['x'])-L_RTL]
        AL = data_Var[data_Var['x']<max(data_Var['x'])-L_RTL]
        AL = AL[AL['x']>L_LTL]
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


    # Hide axis and spines
    if show_axis:
        ax_nrj_diag.get_xaxis().set_visible(True)
        ax_nrj_diag.get_yaxis().set_visible(True)
        ax_nrj_diag.set_xlabel(xaxis_label)
        ax_nrj_diag.set_ylabel(r'Energy [eV]')
    else :
        ax_nrj_diag.get_xaxis().set_visible(False)
        ax_nrj_diag.get_yaxis().set_visible(False)
        for sides in ['right','left','top','bottom']:
            ax_nrj_diag.spines[sides].set_visible(False)
        

    # Legend
    if legend:
        legend_elements = [Line2D([0], [0], color='k', linewidth=line_thick, label='E$_{c}$,E$_{v}$',linestyle='-'),Line2D([0], [0], color='k', linewidth=line_thick, label='E$_{c}$,E$_{v}$',linestyle='--')]
        ax_nrj_diag.legend(handles=legend_elements,loc='upper center',frameon=False,ncol = 2)
    plt.tight_layout()

    # Save file
    if save_fig:
        plt.savefig(fig_name,dpi=300,transparent=True)

def PlotDensSimSS(Var_files,labels,colors=[],num_fig=0,Vext=['nan'],y=['n','p'],xlimits=[],ylimits=[],x_unit='nm',y_unit='cm^-3',plot_type=0,colorbar_type='None',colorbar_display=False,line_type = ['-','--'],legend=True,save_fig=False,fig_name='density.jpg',verbose=True):
    """Make Var_plot for SIMsalabim

    Parameters
    ----------
    Var_files : list
        List of files containing the Var filenames.
    
    labels : list
        List of labels for the Var_files.
    
    colors : list, optional
        List of colors for the Var_files, by default [].

    num_fig : int
        number of the fig to plot JV

    Vext : float
        float to define the voltage at which the densities will be plotted.
        if Vext=['nan'] then takes Vext as max(t), /
        if Vext=['all'] then plots all the Vext,/
        if Vext does not exist we plot the closest voltage, default ['nan']

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
        If True, save density plot as an image with the  file name defined by "fig_name", by default False

    fig_name : str, optional
        name of the file where the figure is saved, by default 'density.jpg'
    
    verbose : bool, optional
        If True, print some information, by default True
    """    
    
    if len(y) > len(line_type):
        if verbose:
            print('\n')
            print('In PlotDensSimSS function')
            print('Invalid line_type list, we meed len(y) == len(line_type)')
            print('We will use default line type instead')
        line_type = []
        for counter, value in enumerate(y):
            line_type.append('-')

    
    # Convert in x-axis
    if x_unit == 'nm':
        xunit_fact = 1e9
        xaxis_label = 'x [nm]'
    elif x_unit == 'um':
        xunit_fact = 1e6
        xaxis_label = 'x [$\mu$m]'
    elif x_unit == 'm':
        xunit_fact = 1
        xaxis_label = 'x [m]'
    else:
        if verbose:
            print('\n')
            print('In PlotDensSimSS function.')
            print('x_unit is wrong so [m] is used.')
        xunit_fact = 1
        xaxis_label = 'x [m]'
    
    # Convert in x-axis
    if y_unit == 'cm^-3':
        yunit_fact = 1e6
        yaxis_label = 'Density [cm$^{-3}$]'
    elif y_unit == 'm^-3':
        yunit_fact = 1
        yaxis_label = 'Density [m$^{-3}$]'
    else:
        if verbose:
            print('\n')
            print('In PlotDensSimSS function.')
            print('y_unit is wrong so [m^-3] is used.')
        yunit_fact = 1
        yaxis_label = 'Density [m$^{-3}$]'
    
    if colors == []:
        colors = ['b','r','g','c','m','y','k']
    
    if (colorbar_type == 'log' or colorbar_type == 'lin') and colorbar_display:
        if verbose:
            print('Line color is set with Vext value.')
    else:
        if verbose:
            print('Line color is set with Var_file name.')
    
    plt.figure(num_fig)
    ax_Vars_plot = plt.gca()

    for Var,lab,c in zip(Var_files,labels,itertools.cycle(colors)):
            
        data_Var = pd.read_csv(Var,delim_whitespace=True) # load data

       
        if Vext[0] == 'nan':
            Vext = [max(data_Var['Vext'])]
            if verbose:
                print('\n')
                print('In PlotDensSimSS function')
                print('V was not specified so Vext = {:.2e} V was plotted'.format(Vext[0]))
        elif Vext[0] == 'all':
            Vext = np.asarray(data_Var['Vext'].unique())
        else:
            pass
            

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
            if verbose:
                print('Wrong colorbar_type input')

        
        for V in Vext:
            data_Var_dum = data_Var[abs(data_Var.Vext -V) == min(abs(data_Var.Vext -V))]
            data_Var_dum = data_Var_dum.reset_index(drop=True)
            if min(abs(data_Var.Vext -V)) != 0:
                if verbose:
                    print('Vext = {:.2e} V was not found so {:.2e} V was plotted'.format(V,data_Var_dum['Vext'][0]))
            
            if (colorbar_type == 'log' or colorbar_type == 'lin') and colorbar_display:
                colorline = s_m.to_rgba(V)
            else:
                colorline = c


            put_label = True
            for i,line in zip(y,line_type):
                if put_label:
                    lab = lab
                    put_label = False
                else:
                    lab = ''

                ax_Vars_plot.plot(data_Var_dum['x']*xunit_fact,data_Var_dum[i]/yunit_fact,color=colorline,label=lab,linestyle=line)
            
    
    # legend
    if legend == True:
        plt.legend(loc='best',frameon=False)
    
    # Add colorbar if needed
    if (colorbar_type == 'log' or colorbar_type == 'lin') and colorbar_display:
        cbar = plt.colorbar(s_m)
        cbar.set_label('Vext [V]')

    # Plot settings
    if plot_type == 1:
        ax_Vars_plot.set_xscale('log')
    elif plot_type == 2:
        ax_Vars_plot.set_yscale('log')
    elif plot_type == 3:
        ax_Vars_plot.set_xscale('log')
        ax_Vars_plot.set_yscale('log')
    else:
        pass

    # legend
    if legend == True:
        if not ((colorbar_type == 'log' or colorbar_type == 'lin') and colorbar_display):
            first_legend = plt.legend(loc='upper center',frameon=False)
            plt.gca().add_artist(first_legend)  
        # plt.legend(handles=[line2], loc='lower right')
        legend_elements = []
        if 'n' in y:
            legend_elements.append(Line2D([0], [0], color='k', label='Electron',linestyle=line_type[int(y.index('n'))]))
        if 'p' in y:
            legend_elements.append(Line2D([0], [0], color='k', label='Hole',linestyle=line_type[int(y.index('p'))]))
        if 'nion' in y:
            legend_elements.append(Line2D([0], [0], color='k', label='Anion',linestyle=line_type[int(y.index('CPI'))]))
        if 'pion' in y:
            legend_elements.append(Line2D([0], [0], color='k', label='Cation',linestyle=line_type[int(y.index('CNI'))]))

        plt.legend(handles=legend_elements,loc='lower center',frameon=False,ncol = 2)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)

    # Set axis limits
    if xlimits != []:
        plt.xlim(xlimits)
    if ylimits != []:
        plt.ylim(ylimits)
    plt.grid(visible=True,which='both')
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_name,dpi=300,transparent=True)

def PlotDensWithTime(Var_files,labels,colors=[],num_fig=0,time=['nan'],y=['n','p'],xlimits=[],ylimits=[],x_unit='s',y_unit='cm^-3', t_unit='s',plot_type=0,colorbar_type='None',colorbar_display=False,line_type = ['-','--'],legend=True,save_fig=False,fig_name='density.jpg',verbose=True):
    """Make Var_plot for SIMsalabim

    Parameters
    ----------
    Var_files : list
        List of files containing the Var filenames.
    
    labels : list
        List of labels for the Var_files.
    
    colors : list, optional
        List of colors for the Var_files, by default [].

    num_fig : int
        number of the fig to plot JV

    time : float
        float to define the voltage at which the densities will be plotted.
        if time=['nan'] then takes time as max(time), /
        if time=['all'] then plots all the time,/
        if time does not exist we plot the closest voltage, default ['nan']

    y : list of str, optional
        yaxis data can be multiple like ['n','p'], by default ['n','p']

    xlimits : list, optional
        x axis limits if = [] it lets python chose limits, by default []

    ylimits : list, optional
        y axis limits if = [] it lets python chose limits, by default []

    x_unit : str, optional
        specify unit of the x-axis either ['ns','us','ms','s'], by default 's'

    y_unit : str, optional
        specify unit of the y-axis either ['cm^-3','m^-3'], by default 'cm^-3'
    
    t_unit : str, optional
        specify unit of the time either ['ns','us','ms','s'], by default 's'

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
        If True, save density plot as an image with the  file name defined by "fig_name", by default False

    fig_name : str, optional
        name of the file where the figure is saved, by default 'density.jpg'
    
    verbose : bool, optional
        If True, print some information, by default True
    """    
    
    if len(y) > len(line_type):
        if verbose:
            print('\n')
            print('In PlotDensWithTime function')
            print('Invalid line_type list, we meed len(y) == len(line_type)')
            print('We will use default line type instead')
        line_type = []
        for counter, value in enumerate(y):
            line_type.append('-')

    
    # Convert in x-axis
    if x_unit == 'nm':
        xunit_fact = 1e9
        xaxis_label = 'x [nm]'
    elif x_unit == 'um':
        xunit_fact = 1e6
        xaxis_label = 'x [$\mu$m]'
    elif x_unit == 'm':
        xunit_fact = 1
        xaxis_label = 'x [m]'
    else:
        if verbose:
            print('\n')
            print('In PlotDensWithTime function.')
            print('x_unit is wrong so [m] is used.')
        xunit_fact = 1
        xaxis_label = 'x [m]'
    
    # Convert in y-axis
    if y_unit == 'cm^-3':
        yunit_fact = 1e6
        yaxis_label = 'Density [cm$^{-3}$]'
    elif y_unit == 'm^-3':
        yunit_fact = 1
        yaxis_label = 'Density [m$^{-3}$]'
    else:
        if verbose:
            print('\n')
            print('In PlotDensWithTime function.')
            print('y_unit is wrong so [m^-3] is used.')
        yunit_fact = 1
        yaxis_label = 'Density [m$^{-3}$]'

    # Convert in time
    if t_unit == 'ns':
        tunit_fact = 1e9
        tunit_label = 'Time [ns]'
    elif t_unit == 'us':
        tunit_fact = 1e6
        tunit_label = 'Time [$\mu$s]'
    elif t_unit == 'ms':
        tunit_fact = 1e3
        tunit_label = 'Time [ms]'
    elif t_unit == 's':
        tunit_fact = 1
        tunit_label = 'Time [s]'
    else:
        if verbose:
            print('\n')
            print('In PlotDensWithTime function.')
            print('t_unit is wrong so [s] is used.')
        tunit_fact = 1
        tunit_label = 'Time [s]'
    
    if colors == []:
        colors = ['b','r','g','c','m','y','k']
    
    if (colorbar_type == 'log' or colorbar_type == 'lin') and colorbar_display:
        if verbose:
            print('Line color is set with time value.')
    else:
        if verbose:
            print('Line color is set with Var_file name.')

    for Var,lab,c in zip(Var_files,labels,itertools.cycle(colors)):
            
        data_Var = pd.read_csv(Var,delim_whitespace=True) # load data

       
        if time[0] == 'nan':
            time = [max(data_Var['time'])]
            if verbose:
                print('\n')
                print('In PlotDensWithTime function')
                print('time was not specified so time = {:.2e} s was plotted'.format(time[0]))
        elif time[0] == 'all':
            time = np.asarray(data_Var['time'].unique())
        else:
            pass

        
        # Prepare the colorbar is there is any
        if colorbar_type == 'log':
            time_bar = data_Var['time']*tunit_fact 
            time_bar = np.asarray(time_bar.drop_duplicates())
            norm = mp.colors.LogNorm(vmin=np.min(time_bar[1]),vmax=np.max(time_bar))
            c_m = mp.cm.viridis# choose a colormap
            s_m = mp.cm.ScalarMappable(cmap=c_m, norm=norm)# create a ScalarMappable and initialize a data structure
            s_m.set_array([])
        elif colorbar_type == 'lin':
            time_bar = data_Var['time']*tunit_fact 
            time_bar = np.asarray(time_bar.drop_duplicates())
            norm = mp.colors.Normalize(vmin=np.min(time_bar),vmax=np.max(time_bar))
            c_m = mp.cm.viridis# choose a colormap
            s_m = mp.cm.ScalarMappable(cmap=c_m, norm=norm) # create a ScalarMappable and initialize a data structure
            s_m.set_array([])
        elif colorbar_type == 'None':
            pass
        else:
            if verbose:
                print('Wrong colorbar_type input')

        plt.figure(num_fig)
        ax_Vars_plot = plt.gca()

        for t in time:
            data_Var_dum = data_Var[abs(data_Var.time -t) == min(abs(data_Var.time -t))]
            data_Var_dum = data_Var_dum.reset_index(drop=True)
            if min(abs(data_Var.time -t)) != 0:
                if verbose:
                    print('time = {:.2e} s was not found so {:.2e} s was plotted'.format(V,data_Var_dum['time'][0]))
            
            if (colorbar_type == 'log' or colorbar_type == 'lin') and colorbar_display:
                colorline = s_m.to_rgba(t*tunit_fact )
            else:
                colorline = c


            put_label = True
            for i,line in zip(y,line_type):
                if put_label:
                    lab = lab
                    put_label = False
                else:
                    lab = ''

                ax_Vars_plot.plot(data_Var_dum['x']*xunit_fact,data_Var_dum[i]/yunit_fact,color=colorline,label=lab,linestyle=line)
            
    
    # legend
    if legend == True:
        plt.legend(loc='best',frameon=False)
    
    # Add colorbar if needed
    if (colorbar_type == 'log' or colorbar_type == 'lin') and colorbar_display:
        cbar = plt.colorbar(s_m)
        cbar.set_label(tunit_label)

    # Plot settings
    if plot_type == 1:
        ax_Vars_plot.set_xscale('log')
    elif plot_type == 2:
        ax_Vars_plot.set_yscale('log')
    elif plot_type == 3:
        ax_Vars_plot.set_xscale('log')
        ax_Vars_plot.set_yscale('log')
    else:
        pass

    # legend
    if legend == True:
        if not ((colorbar_type == 'log' or colorbar_type == 'lin') and colorbar_display):
            first_legend = plt.legend(loc='lower center',frameon=False)
            plt.gca().add_artist(first_legend)  
        # plt.legend(handles=[line2], loc='lower right')
        legend_elements = []
        if 'n' in y:
            legend_elements.append(Line2D([0], [0], color='k', label='Electron',linestyle=line_type[int(y.index('n'))]))
        if 'p' in y:
            legend_elements.append(Line2D([0], [0], color='k', label='Hole',linestyle=line_type[int(y.index('p'))]))
        if 'nion' in y:
            legend_elements.append(Line2D([0], [0], color='k', label='Anion',linestyle=line_type[int(y.index('CPI'))]))
        if 'pion' in y:
            legend_elements.append(Line2D([0], [0], color='k', label='Cation',linestyle=line_type[int(y.index('CNI'))]))

        plt.legend(handles=legend_elements,loc='upper center',frameon=False,ncol = 2)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)

    # Set axis limits
    if xlimits != []:
        plt.xlim(xlimits)
    if ylimits != []:
        plt.ylim(ylimits)
    plt.grid(visible=True,which='both')
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_name,dpi=300,transparent=True)