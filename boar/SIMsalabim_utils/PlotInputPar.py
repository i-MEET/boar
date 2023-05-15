##########################################################################
#################### Read and get parameters from    #####################
#################### dev_para file and command string ####################
##########################################################################
# Author: Vincent M. Le Corre
# Github: https://github.com/VMLC-PV

# Import libraries
import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
# Import SIMsalabim_utils
from boar.SIMsalabim_utils.MakeDevParFile import *
from boar.SIMsalabim_utils.GetInputPar import *

def plot_input_mob(ParFileDic,ax,x_unit='nm',y_unit='m'):
    """Plot the input mobility from the device_parameters.txt

    Parameters
    ----------
    ParFileDic : dic
        dictionary of the device_parameters.txt file
    ax : axes
        axes object where the plot will be done
    x_unit : str, optional
        unit to plot the x-axis, can be ['nm','um','m'], by default 'nm'
    y_unit : str, optional
        unit to plot the y-axis, can be ['cm','m'], by default 'm'
    """  
    ax = ax or plt.gca()
    line_thick = 2
    # Color control for the different layers
    color_nTL = '#c7e6a3'
    color_pTL = '#8cb5f0'
    color_ac = '#ba0000'
    # color_electrode ='#999999'
    color_electrode ='k'

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
        xunit_fact = 1
        xaxis_label = 'x [m]'
        print('\n')
        print('In SIMsalabim_dens_plot function.')
        print('x_unit is wrong so [m] is used. ')
    
    # Convert in y-axis
    if y_unit == 'cm':
        yunit_fact = 1e9
        yaxis_label = '$\mu$ [cm$^2$ V$^{-1}$ s$^{-1}$]'
    elif y_unit == 'm':
        yunit_fact = 1
        yaxis_label = '$\mu$ [m$^2$ V$^{-1}$ s$^{-1}$]'
    else:
        yunit_fact = 1
        yaxis_label = '$\mu$ [m$^2$ V$^{-1}$ s$^{-1}$]'
        print('\n')
        print('In plot_input_mob function.')
        print('y_unit is wrong so [m] is used. ')

    L_LTL = float(ParFileDic['L_LTL'])*xunit_fact
    L_RTL = float(ParFileDic['L_RTL'])*xunit_fact
    L = float(ParFileDic['L'])*xunit_fact
    Lac = L - L_LTL - L_RTL
    mob_LTL = float(ParFileDic['mob_LTL'])
    mob_RTL = float(ParFileDic['mob_RTL'])
    mun_0 = float(ParFileDic['mun_0'])
    mup_0 = float(ParFileDic['mup_0'])

    # Plotting
    # plt.figure(num_fig)
    # ax = plt.axes()
    if L_LTL > 0:
        ax.axvspan(0,L_LTL, alpha=0.5, color=color_nTL)
        ax.plot([0,L_LTL],[mob_LTL,mob_LTL], color = 'k', linewidth = line_thick)
    ax.axvspan(L_LTL,Lac+L_LTL,color=color_ac, alpha=0.5)
    ax.plot([L_LTL,Lac+L_LTL],[mun_0,mun_0], color = 'k', linewidth = line_thick, label = '$\mu_n$')
    ax.plot([L_LTL,Lac+L_LTL],[mup_0,mup_0], color = 'k', linewidth = line_thick, linestyle = 'dashed', label = '$\mu_p$')
    if L_RTL > 0:
        ax.axvspan(Lac+L_LTL,Lac+L_LTL+L_RTL,color=color_pTL, alpha=0.5)
        ax.plot([Lac+L_LTL,Lac+L_LTL+L_RTL],[mob_RTL,mob_RTL], color = 'k', linestyle = 'dashed', linewidth = line_thick)
   
    ax.set_yscale('log')
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)
    ax.legend()



def plot_input_dens(ParFileDic,ax,dens=['Bulk_tr'],x_unit='nm',y_unit='m',y2_unit='m'):
    """Plot the input mobility from the device_parameters.txt

    Parameters
    ----------
    ParFileDic : dic
        dictionary of the device_parameters.txt file
    ax : axes
        axes object where the plot will be done
    dens : list, optional
        list of the densities type to plot, can be ['Bulk_tr','CNI','CPI','St_L','St_R'], by default ['Bulk_tr']
    x_unit : str, optional
        unit to plot the x-axis, can be ['nm','um','m'], by default 'nm'
    y_unit : str, optional
        unit to plot the y-axis, can be ['cm','m'], by default 'm'
    y_unit : str, optional
        unit to plot the y2-axis, can be ['cm','m'], by default 'm'
    """  
    ax = ax or plt.gca()
    ax2 = ax.twinx() # instantiate a second axes that shares the same x-axis
    line_thick = 2
    # Color control for the different layers
    color_nTL = '#c7e6a3'
    color_pTL = '#8cb5f0'
    color_ac = '#ba0000'
    # color_electrode ='#999999'
    color_electrode ='k'

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
        xunit_fact = 1
        xaxis_label = 'x [m]'
        print('\n')
        print('In SIMsalabim_dens_plot function.')
        print('x_unit is wrong so [m] is used. ')
    
    # Convert in y-axis
    if y_unit == 'cm':
        yunit_fact = 1e9
        yaxis_label = 'Density [cm$^{-3}$]'
    elif y_unit == 'm':
        yunit_fact = 1
        yaxis_label = 'Density [m$^{-3}$]'
    else:
        yunit_fact = 1
        yaxis_label = 'Density [cm$^{-3}$]'
        print('\n')
        print('In plot_input_dens function.')
        print('y_unit is wrong so [m] is used. ')
    
    if y2_unit == 'cm':
        y2unit_fact = 1e4
        y2axis_label = 'Surface Trap Density [cm$^{-2}$]'
    elif y2_unit == 'm':
        y2unit_fact = 1
        y2axis_label = 'Surface Trap Density [m$^{-2}$]'
    else:
        y2unit_fact = 1
        y2axis_label = 'Surface Trap Density [m$^{-2}$]'
        print('\n')
        print('In plot_input_lifetime function.')
        print('y2_unit is wrong so [s] is used. ')

    L_LTL = float(ParFileDic['L_LTL'])*xunit_fact
    L_RTL = float(ParFileDic['L_RTL'])*xunit_fact
    L = float(ParFileDic['L'])*xunit_fact
    Lac = L - L_LTL - L_RTL
    Bulk_tr = float(ParFileDic['Bulk_tr'])
    CNI = float(ParFileDic['CNI'])
    CPI = float(ParFileDic['CPI'])
    St_L = float(ParFileDic['St_L'])
    St_R = float(ParFileDic['St_R'])
    TLsTrap = int(ParFileDic['TLsTrap'])
    IonsInTLs = int(ParFileDic['IonsInTLs'])

    if Bulk_tr > 0 or St_L > 0 or St_R > 0 or CNI > 0 or CPI > 0:
        # Plotting
        if L_LTL > 0:
            ax.axvspan(0,L_LTL, color=color_nTL, alpha=0.5)
            if TLsTrap == 1 and 'Bulk_tr' in dens and Bulk_tr > 0:
                ax.plot([0,L_LTL],[Bulk_tr,Bulk_tr], color = 'k', linewidth = line_thick)
            if IonsInTLs == 1:
                if 'CNI' in dens and CNI > 0:
                    ax.plot([0,L_LTL],[CNI,CNI], color = 'k', linewidth = line_thick, linestyle = 'dashed')
                if 'CPI' in dens and CPI > 0:
                    ax.plot([0,L_LTL],[CPI,CPI], color = 'k', linewidth = line_thick, linestyle = 'dotted')
            if St_L > 0:
                ax2.plot([L_LTL],[St_L], color = 'k', marker="x", markersize=10, label = 'St$_L$')
    


        ax.axvspan(L_LTL,Lac+L_LTL,color=color_ac,alpha=0.5)
        if 'Bulk_tr' in dens and Bulk_tr > 0:
            ax.plot([L_LTL,Lac+L_LTL],[Bulk_tr,Bulk_tr], color = 'k', linewidth = line_thick, label = 'Bulk Traps')
        if 'CNI' in dens and CNI > 0:
            ax.plot([L_LTL,Lac+L_LTL],[CNI,CNI], color = 'k', linewidth = line_thick, linestyle = 'dashed', label = 'Anion')
        if 'CPI' in dens and CPI > 0:
            ax.plot([L_LTL,Lac+L_LTL],[CPI,CPI], color = 'k', linewidth = line_thick, linestyle = 'dotted', label = 'Cation')

        if L_RTL > 0:
            ax.axvspan(Lac+L_LTL,Lac+L_LTL+L_RTL,color=color_pTL,alpha=0.5)
            if TLsTrap == 1 and 'Bulk_tr' in dens and Bulk_tr > 0:
                ax.plot([Lac+L_LTL,Lac+L_LTL+L_RTL],[Bulk_tr,Bulk_tr], color = 'k', linewidth = line_thick)
            if IonsInTLs == 1:
                if 'CNI' in dens and CNI > 0:
                    ax.plot([Lac+L_LTL,Lac+L_LTL+L_RTL],[CNI,CNI], color = 'k', linewidth = line_thick, linestyle = 'dashed')
                if 'CPI' in dens and CPI > 0:
                    ax.plot([Lac+L_LTL,Lac+L_LTL+L_RTL],[CPI,CPI], color = 'k', linewidth = line_thick, linestyle = 'dotted')
            if St_R > 0:
                ax2.plot([Lac+L_LTL],[St_R], color = 'k', marker="o", markersize=10, label = 'St$_R$')
            
    
        ax.set_yscale('log')
        ax.set_xlabel(xaxis_label)
        ax.set_ylabel(yaxis_label)
        # ax.legend()
        ax2.set_ylabel(y2axis_label)
        ax2.set_yscale('log')
        legend_elements = [Line2D([0], [0], color='k', linewidth=line_thick, label='Bulk Traps'),Line2D([0], [0], color='k', linewidth=line_thick, linestyle='dashed' ,label='Anion'),Line2D([0], [0], color='k', linewidth=line_thick, linestyle='dotted', label='Cation')]
        if (St_L > 0 and L_LTL > 0) :
            legend_elements.append(Line2D([0], [0], color='k', marker="x", markersize=10, label='St$_L$',linestyle='None'))
        if (St_R > 0 and L_RTL > 0) :
            legend_elements.append(Line2D([0], [0], color='k', marker="o", markersize=10, label='St$_R$',linestyle='None'))
        ax2.legend(handles=legend_elements,ncol=2,loc = 'best')
    else:
        ax.text(0.5, 0.5, 'No Traps or Ions', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)



def plot_input_nrj_diag(ParFileDic,ax,x_unit='nm'):
    """Plot the input energy diagram from the device_parameters.txt

    Parameters
    ----------
    ParFileDic : dic
        dictionary of the device_parameters.txt file
    ax : axes
        axes object where the plot will be done
    x_unit : str, optional
        unit to plot the x-axis, can be ['nm','um','m'], by default 'nm'
    """ 
    ax = ax or plt.gca()
    line_thick = 1
    # Color control for the different layers
    color_nTL = '#c7e6a3'
    color_pTL = '#8cb5f0'
    color_ac = '#ba0000'
    # color_electrode ='#999999'
    color_electrode ='k'

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
        print('\n')
        print('In plot_input_nrj_diag function.')
        print('x_unit is wrong so ''m'' is used. ')

    L_LTL = float(ParFileDic['L_LTL'])*xunit_fact
    L_RTL = float(ParFileDic['L_RTL'])*xunit_fact
    L = float(ParFileDic['L'])*xunit_fact
    Lac = L - L_LTL - L_RTL
    CB = -float(ParFileDic['CB'])
    VB = -float(ParFileDic['VB'])
    CB_LTL = -float(ParFileDic['CB_LTL'])
    CB_RTL = -float(ParFileDic['CB_RTL'])
    VB_LTL = -float(ParFileDic['VB_LTL'])
    VB_RTL = -float(ParFileDic['VB_RTL'])
    W_L = -float(ParFileDic['W_L'])
    W_R = -float(ParFileDic['W_R'])

    # Plotting
    # plt.figure(num_fig)
    # ax = plt.axes()
    if L_LTL > 0:
        ax.fill_between([0,L_LTL],[CB_LTL,CB_LTL],y2=[VB_LTL,VB_LTL],color=color_nTL)
    ax.fill_between([L_LTL,Lac+L_LTL],[CB,CB],y2=[VB,VB],color=color_ac)
    if L_RTL > 0:
        ax.fill_between([Lac+L_LTL,Lac+L_LTL+L_RTL],[CB_RTL,CB_RTL],y2=[VB_RTL,VB_RTL],color=color_pTL)
    ax.plot([-0.1*L,0],[W_L,W_L],color=color_electrode )
    ax.plot([L,L+0.1*L],[W_R,W_R],color=color_electrode )
    

    if float(ParFileDic['Bulk_tr']) > 0:
        ETrapSingle = -float(ParFileDic['ETrapSingle'])
        if int(ParFileDic['TLsTrap']) == 1:
            ax.plot([0,L],[ETrapSingle,ETrapSingle ], color = 'k', linestyle='dashed')
        else:
            ax.plot([L_LTL,Lac+L_LTL],[ETrapSingle,ETrapSingle ], color = 'k',linestyle='dashed')
    if float(ParFileDic['St_L']) > 0:
        ETrapSingle = -float(ParFileDic['ETrapSingle'])
        ax.plot([L_LTL],[ETrapSingle], color = 'k', marker="x", markersize=10)
    if float(ParFileDic['St_R']) > 0:
        ETrapSingle = -float(ParFileDic['ETrapSingle'])
        ax.plot([L_LTL+Lac],[ETrapSingle], color = 'k', marker="x", markersize=10)
        



    # Set the axis      
    ax.set_axisbelow(True)
    ax.grid(visible=True,which='both', linestyle='dashed')#.yaxis.grid(color='gray', linestyle='dashed')
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel('Energy [eV]')
    return



def plot_input_SRH_lifetime(ParFileDic,ax,dens=['Bulk_tr'],x_unit='nm',y_unit='s',y2_unit='m'):
    """Plot the input mobility from the device_parameters.txt

    Parameters
    ----------
    ParFileDic : dic
        dictionary of the device_parameters.txt file
    ax : axes
        axes object where the plot will be done
    dens : list, optional
        list of the densities type to plot, can be ['Bulk_tr','CNI','CPI','St_L','St_R'], by default ['Bulk_tr']
    x_unit : str, optional
        unit to plot the x-axis, can be ['nm','um','m'], by default 'nm'
    y_unit : str, optional
        unit to plot the y-axis, can be ['nm','um','s'], by default 's'
    y2_unit : str, optional
        unit to plot the y2-axis, can be ['ns','us','s'], by default 'm'
    """  
    ax = ax or plt.gca()
    ax2 = ax.twinx() # instantiate a second axes that shares the same x-axis
    line_thick = 2
    # Color control for the different layers
    color_nTL = '#c7e6a3'
    color_pTL = '#8cb5f0'
    color_ac = '#ba0000'
    # color_electrode ='#999999'
    color_electrode ='k'

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
        xunit_fact = 1
        xaxis_label = 'x [m]'
        print('\n')
        print('In SIMsalabim_dens_plot function.')
        print('x_unit is wrong so [m] is used. ')
    
    # Convert in y-axis
    if y_unit == 'ns':
        yunit_fact = 1e9
        yaxis_label = 'Lifetime [ns]'
    elif y_unit == 'us':
        yunit_fact = 1
        yaxis_label = 'Lifetime [$\mu$s]'
    elif y_unit == 's':
        yunit_fact = 1
        yaxis_label = 'Lifetime [s]'
    else:
        yunit_fact = 1
        yaxis_label = 'Density [cm$^{-3}$]'
        print('\n')
        print('In plot_input_lifetime function.')
        print('y_unit is wrong so [s] is used. ')
    
    if y2_unit == 'cm':
        y2unit_fact = 1e2
        y2axis_label = 'Surface rec. velocities [cm s$^{-1}$]'
    elif y2_unit == 'm':
        y2unit_fact = 1
        y2axis_label = 'Surface rec. velocities [m s$^{-1}$]'
    else:
        y2unit_fact = 1
        y2axis_label = 'Surface rec. velocities [m s$^{-1}$]'
        print('\n')
        print('In plot_input_lifetime function.')
        print('y2_unit is wrong so [s] is used. ')

    L_LTL = float(ParFileDic['L_LTL'])*xunit_fact
    L_RTL = float(ParFileDic['L_RTL'])*xunit_fact
    L = float(ParFileDic['L'])*xunit_fact
    Lac = L - L_LTL - L_RTL
    Bulk_tr = float(ParFileDic['Bulk_tr'])
    St_L = float(ParFileDic['St_L'])
    St_R = float(ParFileDic['St_R'])
    TLsTrap = int(ParFileDic['TLsTrap'])
    Cn = float(ParFileDic['Cn'])
    Cp = float(ParFileDic['Cp'])
    if Bulk_tr > 0:
        tau_n = 1/(Bulk_tr*Cn)*yunit_fact
        tau_p = 1/(Bulk_tr*Cp)*yunit_fact
    Sn_R = St_R*Cn*y2unit_fact
    Sn_L = St_L*Cn*y2unit_fact
    Sp_R = St_R*Cp*y2unit_fact
    Sp_L = St_L*Cp*y2unit_fact

    if Bulk_tr > 0 or St_L > 0 or St_R > 0:
        # Plotting
        if L_LTL > 0:
            ax.axvspan(0,L_LTL, color=color_nTL, alpha=0.5)
            if TLsTrap == 1 and 'Bulk_tr' in dens and Bulk_tr > 0:
                ax.plot([0,L_LTL],[tau_n,tau_n], color = 'k', linewidth = line_thick)
                ax.plot([0,L_LTL],[tau_p,tau_pp], color = 'k', linewidth = line_thick, linestyle = 'dashed')
            
            if St_L > 0:
                ax2.plot([L_LTL],[Sn_L], color = 'k', marker="x", markersize=10, label = 'S$_n$')
                ax2.plot([L_LTL],[Sp_L], color = 'k', marker="o", markersize=10, label = 'S$_p$')



        ax.axvspan(L_LTL,Lac+L_LTL,color=color_ac,alpha=0.5)
        if 'Bulk_tr' in dens and Bulk_tr > 0:
            ax.plot([L_LTL,Lac+L_LTL],[tau_n,tau_n], color = 'k', linewidth = line_thick, label = '$\\tau_n$')
            ax.plot([L_LTL,Lac+L_LTL],[tau_p,tau_p], color = 'k', linewidth = line_thick, label = '$\\tau_p$', linestyle = 'dashed')


        if L_RTL > 0:
            ax.axvspan(Lac+L_LTL,Lac+L_LTL+L_RTL,color=color_pTL,alpha=0.5)
            if TLsTrap == 1 and 'Bulk_tr' in dens and Bulk_tr > 0:
                ax.plot([Lac+L_LTL,Lac+L_LTL+L_RTL],[tau_n,tau_n], color = 'k', linewidth = line_thick)
                ax.plot([Lac+L_LTL,Lac+L_LTL+L_RTL],[tau_p,tau_p], color = 'k', linewidth = line_thick, linestyle = 'dashed')
            if St_R > 0:
                ax2.plot([Lac+L_LTL],[Sn_R], color = 'k', marker="x", markersize=10, label = 'S$_n$')
                ax2.plot([Lac+L_LTL],[Sp_R], color = 'k', marker="o", markersize=10, label = 'S$_p$')
        
    
        # Set the axis    
        ax.set_yscale('log')
        ax.set_xlabel(xaxis_label)
        ax.set_ylabel(yaxis_label)
        # ax.legend()
        ax2.set_yscale('log')
        ax2.set_ylabel(y2axis_label)
        legend_elements = [Line2D([0], [0], color='k', linewidth=line_thick, label='$\\tau_n$'),Line2D([0], [0], color='k', linewidth=line_thick, linestyle='dashed' ,label='$\\tau_p$')] 
        if (St_L > 0 and L_LTL > 0) or (St_R > 0 and L_RTL > 0):
            legend_elements.append(Line2D([0], [0], color='k', marker="x", markersize=10, label='S$_n$',linestyle='None'))
            legend_elements.append(Line2D([0], [0], color='k', marker="o", markersize=10, label='S$_p$',linestyle='None'))
        ax2.legend(handles=legend_elements,ncol=2,loc = 'best')
    else:
        ax.text(0.5, 0.5, 'No Traps', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

def VisualizeDevParFile(ParFileDic,num_fig = 0):
    """Visualize the parameters in the ParFileDic.

    Parameters
    ----------
    ParFileDic : dic
        Dictionary of the parameters in the ParFile.
    num_fig : int, optional
        figure number , by default 0
    """    

    fig, axs = plt.subplots(2,2,figsize = (10,8),num = num_fig)
    plot_input_nrj_diag(ParFileDic,ax=axs[0, 0])
    plot_input_mob(ParFileDic,ax=axs[0, 1])
    plot_input_dens(ParFileDic,ax=axs[1, 0])
    plot_input_SRH_lifetime(ParFileDic,ax=axs[1, 1],y_unit='ns',y2_unit='cm')
    plt.tight_layout()