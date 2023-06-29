##########################################################################
##################### Calculate figure-of-merits    #####################
##########################################################################
# by Vincent M. Le Corre
# Package import
import os
import numpy as np
from scipy import constants
# Import homemade package by VLC
# from VLC_units.ManagePlotInputFile.GetInputPar import *
from boar.SIMsalabim_utils.GetInputPar import *


## Physics constants
q = constants.value(u'elementary charge')
eps_0 = constants.value(u'electric constant')
kb = constants.value(u'Boltzmann constant in eV/K')

def Theta_B(ParStrDic,ParFileDic):
    """Calculate theta bimolecular from the 'Device_parameters.txt' file
    and input parameters.\n
     
    Theta_B = (Gamma*Gehp*(L**4))/(mun_0*mup_0*(Vint**2))\n

    Theta bimolecular is define in equation (10) of the article below:\n
    Bartesaghi, D., Pérez, I., Kniepert, J. et al. \n
    Competition between recombination and extraction of free charges determines the fill factor of organic solar cells. \n
    Nat Commun 6, 7083 (2015). https://doi.org/10.1038/ncomms8083\n

    Parameters
    ----------
    ParStrDic : dict
        Contains the parameters and values from the command string

    ParFileDic : dict
        Contains the parameters and values from the 'Device_parameters.txt'

    """
    
    # Choose the parameters to use between command string and file
    CB = float(ChosePar('CB',ParStrDic,ParFileDic))
    VB = float(ChosePar('VB',ParStrDic,ParFileDic))
    eps_r = float(ChosePar('eps_r',ParStrDic,ParFileDic))
    mun_0 = float(ChosePar('mun_0',ParStrDic,ParFileDic))
    mup_0 = float(ChosePar('mup_0',ParStrDic,ParFileDic))
    Gehp = float(ChosePar('Gehp',ParStrDic,ParFileDic))
    L = float(ChosePar('L',ParStrDic,ParFileDic))
    L_LTL = float(ChosePar('L_LTL',ParStrDic,ParFileDic))
    L_RTL = float(ChosePar('L_RTL',ParStrDic,ParFileDic))
    kdirect = float(ChosePar('kdirect',ParStrDic,ParFileDic))
    UseLangevin = int(ChosePar('UseLangevin',ParStrDic,ParFileDic))
   
    if UseLangevin == 1:
        Lang_pre = float(ChosePar('Lang_pre',ParStrDic,ParFileDic))
        Gamma = Lang_pre*q*(mun_0+mup_0)/(eps_0*eps_r)
    else:
        Gamma = kdirect
        

    # Calculate additional parameters    
    Vint = (VB-CB)-0.4 # internal voltage (see paper)
    Lac = L - L_LTL - L_RTL # active layer thickness

    #Return theta bimolecular
    return (Gamma*Gehp*(Lac**4))/(mun_0*mup_0*(Vint**2))

def delta_B(ParStrDic,ParFileDic):
    """Calculate delta bimolecular from the 'Device_parameters.txt' file
    and input parameters.\n

    delta_B = (Gamma*(Nc**2))/Gehp\n

    delta bimolecular is define in equation (12) of the article below:\n
    L. J. A. Koster, V. D. Mihailetchi, R. Ramaker, and P. W. M. Blom ,\n
    "Light intensity dependence of open-circuit voltage of polymer:fullerene solar cells",\n
    Appl. Phys. Lett. 86, 123509 (2005) https://doi.org/10.1063/1.1889240\n

    Parameters
    ----------
    ParStrDic : dict
        Contains the parameters and values from the command string

    ParFileDic : dict
        Contains the parameters and values from the 'Device_parameters.txt'

    """

    # Choose the parameters to use between command string and file
    UseLangevin = int(ChosePar('UseLangevin',ParStrDic,ParFileDic))
    if UseLangevin == 1:
        eps_r = float(ChosePar('eps_r',ParStrDic,ParFileDic))
        mun_0 = float(ChosePar('mun_0',ParStrDic,ParFileDic))
        mup_0 = float(ChosePar('mup_0',ParStrDic,ParFileDic))
        Lang_pre = float(ChosePar('Lang_pre',ParStrDic,ParFileDic))
        Gamma = Lang_pre*q*(mun_0+mup_0)/(eps_0*eps_r)
    else:
        kdirect = float(ChosePar('kdirect',ParStrDic,ParFileDic))
        Gamma = kdirect
        
    Nc = float(ChosePar('Nc',ParStrDic,ParFileDic))
    Gehp = float(ChosePar('Gehp',ParStrDic,ParFileDic))

    #Return delta bimolecular
    return (Gamma*(Nc**2))/(Gehp)


def Theta_T(ParStrDic,ParFileDic):
    """Calculate theta trap from the 'Device_parameters.txt' file
    and input parameters.\n
     
    Theta_T = (Bulk_tr*C_eff*(Lac**2))/(mu_eff*Vint)\n

    TO DO:
    try Theta_T = (Bulk_tr*Cn*Cp*Gehp*(Lac**6))/(mun_0**2*mup_0*Vint**3)\n
     
    Parameters
    ----------
    ParStrDic : dict
        Contains the parameters and values from the command string

    ParFileDic : dict
        Contains the parameters and values from the 'Device_parameters.txt'

    

    """
    
    # Choose the parameters to use between command string and file
    CB = float(ChosePar('CB',ParStrDic,ParFileDic))
    VB = float(ChosePar('VB',ParStrDic,ParFileDic))
    mun_0 = float(ChosePar('mun_0',ParStrDic,ParFileDic))
    mup_0 = float(ChosePar('mup_0',ParStrDic,ParFileDic))
    # Gehp = float(ChosePar('Gehp',ParStrDic,ParFileDic))
    L = float(ChosePar('L',ParStrDic,ParFileDic))
    L_LTL = float(ChosePar('L_LTL',ParStrDic,ParFileDic))
    L_RTL = float(ChosePar('L_RTL',ParStrDic,ParFileDic))
    Bulk_tr = float(ChosePar('Bulk_tr',ParStrDic,ParFileDic))
    Cn = float(ChosePar('Cn',ParStrDic,ParFileDic))
    Cp = float(ChosePar('Cp',ParStrDic,ParFileDic))

    # Calculate additional parameters    
    Vint = (VB-CB)-0.4 # internal voltage (see paper)
    Lac = L - L_LTL - L_RTL # active layer thickness
    mu_eff = np.sqrt(mun_0 * mup_0)
    Lac = L - L_LTL - L_RTL
    C_eff = np.sqrt(Cn * Cp)

    #Return theta trap
    return (Bulk_tr*C_eff*(Lac**2))/(mu_eff*Vint)

def Theta_T2(ParStrDic,ParFileDic):
    """Calculate theta trap from the 'Device_parameters.txt' file
    and input parameters.\n
     
    Theta_T = (Bulk_tr*C_eff*(Lac**2))/(mu_eff*Vint)\n

    TO DO:
    try Theta_T = (Bulk_tr*Cn*Cp*Gehp*(Lac**6))/(mun_0**2*mup_0*Vint**3)\n
     
    Parameters
    ----------
    ParStrDic : dict
        Contains the parameters and values from the command string

    ParFileDic : dict
        Contains the parameters and values from the 'Device_parameters.txt'

    

    """
    
    # Choose the parameters to use between command string and file
    CB = float(ChosePar('CB',ParStrDic,ParFileDic))
    VB = float(ChosePar('VB',ParStrDic,ParFileDic))
    mun_0 = float(ChosePar('mun_0',ParStrDic,ParFileDic))
    mup_0 = float(ChosePar('mup_0',ParStrDic,ParFileDic))
    Gehp = float(ChosePar('Gehp',ParStrDic,ParFileDic))
    L = float(ChosePar('L',ParStrDic,ParFileDic))
    L_LTL = float(ChosePar('L_LTL',ParStrDic,ParFileDic))
    L_RTL = float(ChosePar('L_RTL',ParStrDic,ParFileDic))
    Bulk_tr = float(ChosePar('Bulk_tr',ParStrDic,ParFileDic))
    Cn = float(ChosePar('Cn',ParStrDic,ParFileDic))
    Cp = float(ChosePar('Cp',ParStrDic,ParFileDic))

    # Calculate additional parameters    
    Vint = (VB-CB)-0.4 # internal voltage (see paper)
    Lac = L - L_LTL - L_RTL # active layer thickness
    mu_eff = np.sqrt(mun_0 * mup_0)
    Lac = L - L_LTL - L_RTL
    C_eff = np.sqrt(Cn * Cp)

    #Return theta trap
    return (Bulk_tr*Cn*Cp*(Lac**2))/(Gehp*mun_0**2*mup_0**2*Vint**4)
   
   

def delta_T(ParStrDic,ParFileDic):
    """Calculate delta trap from the 'Device_parameters.txt' file
    and input parameters.\n

    delta_T = (Bulk_tr*C_eff*Nc)/(Gehp)\n

    TO DO:
    try delta_T = (Bulk_tr*Cn*Cp*Nc**2)/(Gehp**2)

    Parameters
    ----------
    ParStrDic : dict
        Contains the parameters and values from the command string

    ParFileDic : dict
        Contains the parameters and values from the 'Device_parameters.txt'

    """

    # Choose the parameters to use between command string and file
    
    
    Nc = float(ChosePar('Nc',ParStrDic,ParFileDic))
    Gehp = float(ChosePar('Gehp',ParStrDic,ParFileDic))
    Bulk_tr = float(ChosePar('Bulk_tr',ParStrDic,ParFileDic))
    Cn = float(ChosePar('Cn',ParStrDic,ParFileDic))
    Cp = float(ChosePar('Cp',ParStrDic,ParFileDic))

    # Calculate additional parameters    
    C_eff = np.sqrt(Cn * Cp)

    #Return delta trap
    return (Bulk_tr*C_eff*Nc)/(Gehp)

def w_int_L(ParStrDic,ParFileDic,relative=False):
    """Calculate surface trap FOM from the 'Device_parameters.txt' file
    and input parameters.\n

    w_int = miin(Cn,Cp)*St_L*(Nc_LTL/Nc)*exp((CB_LTL-CB)/kb*T)\n

   
    Parameters
    ----------
    ParStrDic : dict
        Contains the parameters and values from the command string

    ParFileDic : dict
        Contains the parameters and values from the 'Device_parameters.txt'
    
    relative : bool
        If True, CB_LTL is relative to CB
    

    """

    Nc = float(ChosePar('Nc',ParStrDic,ParFileDic))
    Nc_LTL = float(ChosePar('Nc_LTL',ParStrDic,ParFileDic))
    CB = float(ChosePar('CB',ParStrDic,ParFileDic))
    CB_LTL = float(ChosePar('CB_LTL',ParStrDic,ParFileDic))
    if relative:
        if float(ChosePar('L_LTL',ParStrDic,ParFileDic))>0:
            CB_LTL = CB_LTL + CB
        else:
            raise ValueError('L_LTL must be > 0 to use relative CB_LTL')
            

    T = float(ChosePar('T',ParStrDic,ParFileDic))
    Cn = float(ChosePar('Cn',ParStrDic,ParFileDic))
    Cp = float(ChosePar('Cp',ParStrDic,ParFileDic))
    St_L = float(ChosePar('St_L',ParStrDic,ParFileDic))

    # Calculate additional parameters
    Cmin = min(Cn,Cp)
    w_int = Cmin*St_L*(Nc_LTL/Nc)*np.exp((CB_LTL-CB)/(kb*T))

    return w_int

def w_int_R(ParStrDic,ParFileDic,relative=False):
    """Calculate surface trap FOM from the 'Device_parameters.txt' file
    and input parameters.\n

    w_int = miin(Cn,Cp)*St_R*(Nc_RTL/Nc)*exp((VB_RTL-VB)/kb*T)\n

   
    Parameters
    ----------
    ParStrDic : dict
        Contains the parameters and values from the command string

    ParFileDic : dict
        Contains the parameters and values from the 'Device_parameters.txt'
    
    relative : bool
        If True, VB_RTL is relative to VB

    """

    Nc = float(ChosePar('Nc',ParStrDic,ParFileDic))
    Nc_RTL = float(ChosePar('Nc_RTL',ParStrDic,ParFileDic))
    VB = float(ChosePar('VB',ParStrDic,ParFileDic))
    VB_RTL = float(ChosePar('VB_RTL',ParStrDic,ParFileDic))
    if relative:
        if float(ChosePar('L_RTL',ParStrDic,ParFileDic))>0:
            VB_RTL = VB + VB_RTL
        else:
            raise ValueError('L_RTL must be > 0 to use relative VB_RTL')
    T = float(ChosePar('T',ParStrDic,ParFileDic))
    Cn = float(ChosePar('Cn',ParStrDic,ParFileDic))
    Cp = float(ChosePar('Cp',ParStrDic,ParFileDic))
    St_R = float(ChosePar('St_R',ParStrDic,ParFileDic))

    # Calculate additional parameters
    Cmin = min(Cn,Cp)
    w_int = Cmin*St_R*(Nc_RTL/Nc)*np.exp((VB_RTL-VB)/(kb*T))

    return w_int

def Vbi(ParStrDic,ParFileDic,relative=True):
    """Calculate the built-in voltage from the 'Device_parameters.txt' file
    and input parameters.\n

    Vbi = abs(W_L-W_R)\n

    Parameters
    ----------
    ParStrDic : dict
        Contains the parameters and values from the command string

    ParFileDic : dict
        Contains the parameters and values from the 'Device_parameters.txt'
    
    relative : bool
        If True, CB_LTL and VB_RTL are relative to CB and VB respectively and W_L and W_R are relative to CB_LTL and VB_RTL respectively
    """

    # Choose the parameters to use between command string and file
    W_L = float(ChosePar('W_L',ParStrDic,ParFileDic))
    W_R = float(ChosePar('W_R',ParStrDic,ParFileDic))
    CB = float(ChosePar('CB',ParStrDic,ParFileDic))
    VB = float(ChosePar('VB',ParStrDic,ParFileDic))
    CB_LTL = float(ChosePar('CB_LTL',ParStrDic,ParFileDic))
    VB_RTL = float(ChosePar('VB_RTL',ParStrDic,ParFileDic))
    
    if relative:
        if float(ChosePar('L_LTL',ParStrDic,ParFileDic))>0:
            CB_LTL = CB + CB_LTL
        else:
            CB_LTL = CB
        if float(ChosePar('L_RTL',ParStrDic,ParFileDic))>0:
            VB_RTL = VB + VB_RTL
        else:
            VB_RTL = VB

        W_L =  CB_LTL + W_L
        W_R = VB_RTL + W_R

    #Return Vbi
    return abs((W_L-W_R))


def get_Rseries(ParStrDic,ParFileDic):
    """Get the series resistance from the 'Device_parameters.txt' file

    Parameters
    ----------
    ParStrDic : dict
        Contains the parameters and values from the command string

    ParFileDic : dict
        Contains the parameters and values from the 'Device_parameters.txt'
    """


    return float(ChosePar('Rseries',ParStrDic,ParFileDic))

def get_Rshunt(ParStrDic,ParFileDic):
    """Get the shunt resistance from the 'Device_parameters.txt' file

    Parameters
    ----------
    ParStrDic : dict
        Contains the parameters and values from the command string

    ParFileDic : dict
        Contains the parameters and values from the 'Device_parameters.txt'
    """

    return float(ChosePar('Rshunt',ParStrDic,ParFileDic))
