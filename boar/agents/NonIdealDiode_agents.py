## Drift diffusion agent
# Version 0.1
# (c) Larry Lueer, Vincent M. Le Corre, i-MEET 2021-2022

# Import libraries
import sys
import os,itertools
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.special import lambertw
from scipy import constants

# Import boar
from boar import *

# Import homemade package by VLC
from boar.SIMsalabim_utils.RunSim import *
from boar.SIMsalabim_utils.CalcFom import *
# from boar.SIMsalabim_utils.plot_settings_screen import *

## Physics constants
q = constants.value(u'elementary charge')
eps_0 = constants.value(u'electric constant')
kb = constants.value(u'Boltzmann constant in eV/K')

class Non_Ideal_Diode_agent():
    """ Agent to run drift diffusion simulations with SIMsalabim to be used with BOAR MultiObjectiveOptimizer
    
    Parameters
    ----------
    path2simu : str
        Path to the simulation executable

    """
    def __init__(self,path2simu = '') -> None:
        self.path2simu = path2simu

    
    

    
    def NonIdealDiode_dark(self,V, J0, n, Rs, Rsh, T = 300):
        """ Solve non ideal diode equation for dark current
            J = J0*[exp(-(V-J*Rs)/(n*Vt*)) - 1] + (V - J*Rs)/Rsh
            with the method described in:
            Solid-State Electronics 44 (2000) 1861-1864,
            see equation (4)-(5)
        

        Parameters
        ----------
        V : 1-D sequence of floats
            Array containing the voltages.
        J0 : float
            Dark Saturation Current.
        n : float
            Ideality factor.
        Rs : float
            Series resistance.
        Rsh : float
            Shunt resistance.
        T : float, optional
            Absolute temperature , by default 300

        Returns
        -------
        1-D sequence of floats
            Array containing the currents.
        """    
        Vt = kb*T
        w = lambertw(((J0*Rs*Rsh)/(n*Vt*(Rs+Rsh)))*np.exp((Rsh*(V+J0*Rs))/(n*Vt*(Rs+Rsh)))) # check equation (5) in the paper

        Current = (n*Vt/Rs) * w + ((V-J0*Rsh)/(Rs+Rsh))
        return Current.real

    def NonIdealDiode_dark_log(self,V, J0, n, Rs, Rsh, T = 300):
        """ The logarithmic version of the non ideal diode equation for dark current
        see NonIdealDiode_dark for more details

        note: this is only useful when doing the fits of the dark current-voltage characteristics.
            

        Parameters
        ----------
        V : 1-D sequence of floats
            Array containing the voltages.
        J0 : float
            Dark Saturation Current.
        n : float
            Ideality factor.
        Rs : float
            Series resistance.
        Rsh : float
            Shunt resistance.
        T : float, optional
            Absolute temperature , by default 300
        
        Returns
        -------
        1-D sequence of floats
            Array containing the currents.
        """        
        return np.log(abs(self.NonIdealDiode_dark(V, J0, n, Rs, Rsh, T)))

    def NonIdealDiode_light(self,V,J0,n,Rs,Rsh,Jph,T=300):
        """ Solve non ideal diode equation for light current
            J = Jph - J0*[exp(-(V-J*Rs)/(n*Vt*)) - 1] - (V - J*Rs)/Rsh
            with the method described in:
            Solar Energy Materials & Solar Cells 81 (2004) 269–277
            see equation (1)-(2)


        Parameters
        ----------
        V : 1-D sequence of floats
            Array containing the voltages.
        J0 : float
            Dark Saturation Current.
        n : float
            Ideality factor.
        Rs : float
            Series resistance.
        Rsh : float
            Shunt resistance.
        Jph : float
            Photocurrent.
        T : float, optional
            Absolute temperature , by default 300

        Returns
        -------
        1-D sequence of floats
            Array containing the currents.
        """   

        Vt = kb*T
        w = lambertw(((J0*Rs*Rsh)/(n*Vt*(Rs+Rsh)))*np.exp((Rsh*(V+Jph*Rs+J0*Rs))/(n*Vt*(Rs+Rsh)))) # check equation (2) in the paper
        w = w.real # remove the imaginary part
        Current = -(V/(Rs+Rsh)) - (n*Vt/Rs) * w + ((Rsh*(J0+Jph))/(Rs+Rsh))

        return -Current.real



    def NonIdealDiode_light_log(self,V,J0,n,Rs,Rsh,Jph,T=300):
        """ The logarithmic version of the non ideal diode equation for light current
        see NonIdealDiode_light for more details

        note: this is only useful when doing the fits of the light current-voltage characteristics.
            

        Parameters
        ----------
        V : 1-D sequence of floats
            Array containing the voltages.
        J0 : float
            Dark Saturation Current.
        n : float
            Ideality factor.
        Rs : float
            Series resistance.
        Rsh : float
            Shunt resistance.
        Jph : float
            Photocurrent.
        T : float, optional
            Absolute temperature , by default 300
        
        Returns
        -------
        1-D sequence of floats
            Array containing the currents.
        """        
        return np.log(abs(self.NonIdealDiode_light(V,J0,n,Rs,Rsh,Jph,T)))

    def DifferentialResistance(self,V,J):
        """ Calculate the differential resistance of a diode from the voltage and current with:

            Rdiff = dV/dJ

        Parameters
        ----------
        V : 1-D sequence of floats
            Array containing the voltages.
        J : 1-D sequence of floats
            Array containing the currents.

        Returns
        -------
        1-D sequence of floats
            Array containing the differential resistance.
        """    
        # remove idx where np.diff(J) == 0 not to divide by zero
        idx = np.where(np.diff(J) == 0)[0]
        V = np.delete(V,idx)
        J = np.delete(J,idx)

        return np.diff(V) / np.diff(J) 

    def DifferentialIdealityFactor(self,V,J,T=300):
        """ Calculate the differential ideality factor of a diode from the voltage and current with:

            ndiff = 1/(Vt*dV/d(ln(J)))
        
        Parameters
        ----------
        V : 1-D sequence of floats
            Array containing the voltages.
        J : 1-D sequence of floats
            Array containing the currents.
        T : float, optional
            Absolute temperature , by default 300

        Returns
        -------
        1-D sequence of floats
            Array containing the differential ideality factor.
        """    

        Vt = kb*T
        return 1/(Vt*np.diff(np.log(abs(J))) / np.diff(V))

    def get_Jsc(self,Volt,Curr):
        """Get the short-circuit current (Jsc) from solar cell JV-curve by interpolating the current at 0 V

        Parameters
        ----------
        Volt : 1-D sequence of floats
            Array containing the voltages.

        Curr : 1-D sequence of floats
            Array containing the current-densities.

        Returns
        -------
        Jsc : float
            Short-circuit current value
        """
        Jsc_dumb = np.interp(0, Volt, Curr)
        return Jsc_dumb

    def get_Voc(self,Volt,Curr):
        """Get the Open-circuit voltage (Voc) from solar cell JV-curve by interpolating the Voltage when the current is 0

        Parameters
        ----------
        Volt : 1-D sequence of floats
            Array containing the voltages.

        Curr : 1-D sequence of floats
            Array containing the current-densities.

        Returns
        -------
        Voc : float
            Open-circuit voltage value
        """
        Voc_dumb = np.interp(0, Curr, Volt)
        return Voc_dumb

    def FitNonIdealDiode(self,V,J,T=300,JV_type='dark',take_log=True,bounds = ([1e-30, 0.8, 1e-8, 1e-3], [1e-3, 3, 1e2, 1e8]),p_start={}):
        """ Fit the non ideal diode equation to the data using the least squares method.

            see NonIdealDiode_dark and NonIdealDiode_light for more details  

        Parameters
        ----------
        V : 1-D sequence of floats
            Array containing the voltages.
        J : 1-D sequence of floats
            Array containing the currents.
        T : float, optional
            Absolute temperature , by default 300
        JV_type : str, optional
            Type of JV curve to fit. Can be 'dark' or 'light'.
        take_log : bool, optional
            If True, take the logarithm of the current.
        bounds : tuple, optional
            Bounds for the fit. The default is ([1e-20, 0.8, 1e-8, 1e-3], [1e-3, 3, 1e2, 1e8])

        Returns
        -------
        dict
            Dictionary containing the fit results.
            {'J0':J0, 'J0_err':J0_err, 'n':n, 'n_err':n_err, 'Rs':Rs, 'Rs_err':Rs_err, 'Rsh':Rsh, 'Rsh_err':Rsh_err}
        """ 
        Vt = kb*T
        V = np.asarray(V)
        J = np.asarray(J)
        if JV_type == 'dark':
            Rdiff = self.DifferentialResistance(V,J)
            
            # Remove data close to 0 V to get the ideality factor for the exponential region
            V1 = V[V>0.4]
            J2 = J[V>0.4] 
            ndiff = self.DifferentialIdealityFactor(V1,J2,T)
            
            # Try to get good starting point 
            
            n_ = min(ndiff)

            if n_ < bounds[0][1] or n_ > bounds[1][1]:
                n_ = 1.5 # if the ideality factor is outside the bounds, use a default value
            Rs_ = min(Rdiff)
            if Rs_ < bounds[0][2] or Rs_ > bounds[1][2]:
                Rs_ = 1e-3 # if the series resistance is outside the bounds, use a default value
            Rsh_ = max(Rdiff)
            if Rsh_ < bounds[0][3] or Rsh_ > bounds[1][3]:
                Rsh_ = 1e8 # if the shunt resistance is outside the bounds, use a default value
            
            J0_ = min(bounds[0][0]*2,min(abs(J)))   
            bounds[0][0] = J0_*1e-4

            # remove 0V data for some reason it crashes if there is 0V data
            pos_0V = np.where(abs(V) < 1e-10 )[0]
            if len(pos_0V) > 0:
                V = np.delete(V, pos_0V[0])
                J = np.delete(J, pos_0V[0])

            bounds_ = bounds

            if take_log:
                J = np.log(abs(J))
                diode_func = self.NonIdealDiode_dark_log
            else:
                diode_func = self.NonIdealDiode_dark
                
            # check if starting point is given
            if 'J0' in p_start:
                J0_ = p_start['J0']
                bounds_[0][0] = J0_*1e-4
            if 'n' in p_start:
                n_ = p_start['n']
                
            if 'Rs' in p_start:
                Rs_ = p_start['Rs']
                
            if 'Rsh' in p_start:
                Rsh_ = p_start['Rsh']
            p0_ = [J0_, n_ , Rs_, Rsh_]

        elif JV_type == 'light':

            Jsc = abs(self.get_Jsc(V,J))
            Voc = self.get_Voc(V,J)

            # Remove data close to 0 V to get the ideality factor for the exponential region
            V1 = V[V>0.4]
            J2 = J[V>0.4] 
            ndiff = self.DifferentialIdealityFactor(V1,J2,T)
            n_ = min(ndiff)
            
            # Get starting value for Rs from the slope at Voc
            Voc_idx = np.argmin(abs(J)) # Find closest value to Voc in the data
            slope,intercept = np.polyfit([V[Voc_idx-1],V[Voc_idx],V[Voc_idx+1]],[J[Voc_idx-1],J[Voc_idx],J[Voc_idx+1]],1)
            Rs_ = 1/slope

            # Get starting value for Rsh from the slope at Jsc
            Jsc_idx = np.argmin(abs(V)) # Find closest value to Jsc in the data
            slope,intercept = np.polyfit([V[Jsc_idx-1],V[Jsc_idx],V[Jsc_idx+1]],[J[Jsc_idx-1],J[Jsc_idx],J[Jsc_idx+1]],1)
            Rsh_ = 1/slope

            # Get starting value for J0 
            J0_ = abs((Jsc - Voc/Rsh_)*(np.exp(-Voc/(n_*Vt))))

            # Make sure starting value is within the bounds
            if Rs_ < bounds[0][2] or Rs_ > bounds[1][2]:
                Rs_ = 1e-3 # if the series resistance is outside the bounds, use a default value
            if Rsh_ < bounds[0][3] or Rsh_ > bounds[1][3]:
                Rsh_ = 1e8 # if the shunt resistance is outside the bounds, use a default value
            if J0_ < bounds[0][0] or J0_ > bounds[1][0]:
                J0_ = 1e-7
            if n_ < bounds[0][1] or n_ > bounds[1][1]:
                n_ = 2 # if the ideality factor is outside the bounds, use a default value
            bounds_ = list(bounds)
            bounds_[0].append(Jsc*0.1)
            bounds_[1].append(Jsc*10)
            bounds_ = tuple(bounds_)

            # Set the starting values for the fit
            # check if starting point is given
            if 'J0' in p_start:
                J0_ = p_start['J0']
                bounds_[0][0] = J0_*1e-4
            if 'n' in p_start:
                n_ = p_start['n']
                
            if 'Rs' in p_start:
                Rs_ = p_start['Rs']
                
            if 'Rsh' in p_start:
                Rsh_ = p_start['Rsh']

            if 'Jph' in p_start:
                Jph_ = p_start['Jph']
            else:
                Jph_ = Jsc
            
            p0_ = [J0_, n_ , Rs_, Rsh_, Jph_]
            
            # remove 0V data for some reason it crashes if there is 0V data
            pos_0V = np.where(V == 0)[0]
            if len(pos_0V) > 0:
                V = np.delete(V, pos_0V[0])
                J = np.delete(J, pos_0V[0])
            
            # Get the log of the current if needed
            if take_log:
                J = np.log(abs(J))
                diode_func = self.NonIdealDiode_light_log
            else:
                diode_func = self.NonIdealDiode_light

        else:
            raise ValueError('JV_type must be either ''dark'' or ''light''')
        
        # # Fit the non ideal diode equation
        popt, pcov = curve_fit(diode_func, V, J, p0=p0_, maxfev = 5e3,bounds = bounds_, method = 'dogbox')
        perr = np.sqrt(np.diag(pcov)) # error of the fit

        if JV_type == 'dark':
            J0, n, Rs, Rsh = popt
            J0_err, n_err, Rs_err, Rsh_err = perr
            return {'J0':J0, 'J0_err':J0_err, 'n':n, 'n_err':n_err, 'Rs':Rs, 'Rs_err':Rs_err, 'Rsh':Rsh, 'Rsh_err':Rsh_err}
        elif JV_type == 'light':
            J0, n, Rs, Rsh, Jph = popt
            J0_err, n_err, Rs_err, Rsh_err, Jph_err = perr
            return {'J0':J0, 'J0_err':J0_err, 'n':n, 'n_err':n_err, 'Rs':Rs, 'Rs_err':Rs_err, 'Rsh':Rsh, 'Rsh_err':Rsh_err, 'Jph':Jph, 'Jph_err':Jph_err}

    

    