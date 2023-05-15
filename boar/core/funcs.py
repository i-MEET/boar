##########################################################################
##########################   Useful functions   ##########################
##########################################################################
# Version 0.1
# (c) Larry Lueer, Vincent M. Le Corre, i-MEET 2021-2023

# Import libraries
import numpy as np
from copy import deepcopy
from functools import partial
from typing import Callable, Any


def get_unique_X(X,xaxis,X_dimensions):
        """Get the unique values of the independent variable (X) in the dataset

        Parameters
        ----------
        X : ndarray
            the experimental dimensions
        xaxis : str, optional
            the name of the independent variable
        X_dimensions : list, optional
            names of the X columns
        Returns
        -------
        X_unique : ndarray
            the unique values of the independent variable
        X_dimensions_uni : list
            the names of the columns of X_unique

        Raises
        ------
        ValueError
            if xaxis is not in X_dimensions

        """
        X_unique = deepcopy(X)
        idx_x = None
        if xaxis in X_dimensions:
            idx_x = X_dimensions.index(xaxis)
        else:
            raise ValueError(xaxis + ' not in X_dimensions, please add it to X_dimensions')

        X_unique = np.delete(X_unique,idx_x,axis=1)
        X_dimensions_uni = [x for x in X_dimensions if x != xaxis]
        # get index of unique values
        unique,idxuni = np.unique(X_unique,axis=0,return_index=True)

        X_unique = X_unique[np.sort(idxuni),:] # resort X_unique 

        return X_unique,X_dimensions_uni

def get_unique_X_and_xaxis_values(X,xaxis,X_dimensions):
        """Get the values of the independent variable (X) in the dataset for each unique value of the other dimensions

        Parameters
        ----------
        X : ndarray
            the experimental dimensions
        xaxis : str, optional
            the name of the independent variable
        X_dimensions : list, optional
            the names of the columns of X
        Returns
        -------
        xs : list of ndarrays
            the values of the independent variable for each unique value of the other dimensions

        """
        X_unique, X_dimensions_uni = get_unique_X(X,xaxis,X_dimensions) # get unique X values and their dimensions
        idx_x = int(X_dimensions.index(xaxis))
        xs = []
        for uni in X_unique:
            
            X_dum = deepcopy(X)
            # drop the xaxis column
            X_dum = np.delete(X_dum,X_dimensions.index(xaxis),axis=1)
            # find indexes where the other columns are equal to the unique values
            idxs = np.where(np.all(X_dum==uni,axis=1))[0]
            # get the values of the xaxis for these indexes
            xs.append(X[idxs,idx_x])
            
            

        return X_unique, X_dimensions_uni, xs

def callable_name(any_callable: Callable[..., Any]) -> str:
    """Returns the name of a callable object

    Parameters
    ----------
    any_callable : Callable[..., Any]
        Callable object

    Returns
    -------
    str
        Name of the callable object
    """    
    if isinstance(any_callable, partial):
        return any_callable.func.__name__
    else:
        try:
            return any_callable.__name__
        except AttributeError:
            return str(any_callable)


def gaussian_pulse_norm(t, tpulse, width):
    """Returns a gaussian pulse

    Parameters
    ----------
    t : 1-D sequence of floats
        t time axis (unit: s)
    tpulse : float
        tpulse center of the pulse (unit: s)
    width : float
        width of the pulse (unit: s)

    Returns
    -------
    1-D sequence of floats
        Vector containing the gaussian pulse
    """    
    return np.exp(-np.power(t - tpulse, 2.) / (2 * np.power(width, 2.)))

def polynom(x,a,gamma):
    return a*(x**gamma)

def sigmoid(x,a,b,xc):
    return a/2*(1+np.tanh((x-xc)/(2*b)))

def gauss(x,a,b,xc):
    return a/(b*np.sqrt(2*np.pi)) * np.exp(-1/2*((x-xc)/b)**2) # b is the standard deviation of a Normal distribution

def gauss_sk(x,a,b,c,s=0.0001): # Gauss function VERIFIED 11 June 2020
    # a= amps, b=width, c=center, s=skewness parameter
    S=a/(b*np.sqrt(2*np.pi))*np.exp(-(np.log(1+np.sqrt(2)/2*s*((x-c)/b))/s)**2)
    S[np.isnan(S)]=0
    return S


def sci_notation(number, sig_fig=2):
    """Make proper scientific notation for graphs

    Parameters
    ----------
    number : float
        Number to put in scientific notation.

    sig_fig : int, optional
        Number of significant digits (Defaults = 2).

    Returns
    -------
    output : str
        String containing the number in scientific notation
    """
    if sig_fig != -1:
        if number == 0:
            output = '0'
        else:
            ret_string = "{0:.{1:d}e}".format(number, sig_fig)
            a,b = ret_string.split("e")
            if int(b) >= 0:
                b = int(b) #removed leading "+" and strips leading zeros too.
                c = ''
            else: 
                b = abs(int(b))
                c = u"\u207B" # superscript minus sign
            SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
            b = str(b).translate(SUP)
            output =a + ' x 10' + c + b
    else:
        if number == 0:
            output = '0'
        else: 
            ret_string = "{0:.{1:d}e}".format(number, 0)
            a,b = ret_string.split("e")
            b = int(b) #removed leading "+" and strips leading zeros too.
            if int(b) >= 0:
                b = int(b) #removed leading "+" and strips leading zeros too.
                c = ''
            else: 
                b = abs(int(b))
                c = u"\u207B" # superscript minus sign
            SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
            #SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
            b = str(b).translate(SUP)
            output = '10' + c + b    
    return output



def get_flux_density(P,wl,nu,A,alpha):
    """
    From the measured power and reprate and area, 
    get photons/cm2 and approximate photons/cm3 per pulse

    Args:
        P (float): total CW power of pulse in W
        wl (float): excitation wavelength in nm
        nu (float): repetition rate in s-1
        A (float): effective pump area in cm2
        alpha (float): penetration depth in cm

    Returns:
        flux (float): flux in photons per cm2
        density (float): average volume density in photons/cm3
    """
     
    E = 1e7/wl/8065 * 1.603e-19 # convert wavelength to J for a single photon
    Epu = P/nu # energy in J of a single pulse
    Nph = Epu/E # Photon number in pulse
    flux = Nph/A # flux in photons / cm2
    density = flux/alpha # average absorbed density in photons/cm3
    return flux,density