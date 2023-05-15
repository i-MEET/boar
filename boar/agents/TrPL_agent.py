######################################################################
################## Transient Absorption agent ########################
######################################################################
# Version 0.1
# (c) Larry Lueer, Vincent M. Le Corre, i-MEET 2021-2023

# Import libraries
import sys
import os,itertools
from scipy import interpolate, constants
from copy import deepcopy
import matplotlib.pyplot as plt
# Import boar
from boar.dynamic_utils.pump import *
from boar.dynamic_utils.rate_eq import *
from boar.agents.Agent import Agent
from boar.core.funcs import callable_name, get_unique_X, get_unique_X_and_xaxis_values

## Physics constants
q = constants.value(u'elementary charge')
kb = constants.value(u'Boltzmann constant in eV/K')

class TrPL_agent(Agent):
    """ Agent to run Transient Phololuminescence simulation based on rate equations to be used with BOAR MultiObjectiveOptimizer
    
    Parameters
    ----------
    trPL_model : function, optional
        trPL model to be used, by default Bimolecular_Trapping_equation
        To see the available models, check the rate_eq.py file in the dynamic_utils folder

    pump_model : function, optional
        pump model to be used, by default square_pump
        To see the available models, check the pump.py file in the dynamic_utils folder

    pump_params : dict, optional
        dictionary of pump parameters, by default {'P':0.0039, 'wvl':850, 'fpu':10000, 'A':0.3*0.3*1e-4, 'alpha':1e-5*1e-2, 'pulse_width':0.2*(1/10000), 't0':0, 'background':0}
        P : float
            total CW power of pulse in W
        wvl : float
            excitation wavelength in nm
        fpu : float
            pump frequency in Hz
        A : float
            effective pump area in m^-2
        alpha : float
            penetration depth in m
        pulse_width : float
            width of the pump pulse in seconds
        t0 : float, optional
            time shift of the pump pulse, by default 0
        background : float, optional
            background volume density of generated photons, by default 0



    """
    def __init__(self,trPL_model = Bimolecular_Trapping_equation,pump_model = initial_carrier_density, pump_params = {'fpu':1000, 'background':0},flux_density_model = get_flux_density) -> None:
        super().__init__()
        self.flux_density_model = flux_density_model
        self.trPL_model = trPL_model
        self.pump_model = pump_model
        self.pump_params= pump_params


        #  check model function name even when it is called inside a partial function
        model_name = callable_name(self.trPL_model)
        # if self.model_name == 'Bimolecular_Trapping_equation':
        if model_name == 'Bimolecular_Trapping_equation':
            self.trPL_default_val = {'kdirect' : 1e-18, 'ktrap': 1e5, 'QE':0.9, 'I_PL': 1e-17} # kwargs for the trPL model  by defaults (tpulse=None, equilibrate=True,eq_limit=1e-2)
        # elif self.trPL_model == Bimolecular_Trapping_Detrapping_equation:
        elif model_name == 'Bimolecular_Trapping_Detrapping_equation':
            self.trPL_default_val = {'kdirect' : 26e-17, 'ktrap': 1.2e-15, 'kdetrap': 80e-17,'Bulk_tr':6e18,'p_0':65e18, 'QE':0.9, 'I_PL': 1e-17} # kwargs for the trPL model  by defaults (tpulse=None, equilibrate=True,eq_limit=1e-2) ktrap = 1.2e-15
        else:
            self.trPL_default_val = {}

        self.model_name = model_name
        # if self.model_name == 'Bimolecular_Trapping_equation':
        #     self.trPL_default_val = {'kdirect' : 1e-18, 'ktrap': 1e5, 'QE':0.9, 'I_PL': 1e-17} # kwargs for the trPL model  by defaults (tpulse=None, equilibrate=True,eq_limit=1e-2)
        # elif self.trPL_model == Bimolecular_Trapping_Detrapping_equation:
        #     self.trPL_default_val = {'kdirect' : 26e-17, 'ktrap': 1.2e-15, 'kdetrap': 80e-17,'Bulk_tr':6e18,'p_0':65e18, 'QE':0.9, 'I_PL': 1e-17} # kwargs for the trPL model  by defaults (tpulse=None, equilibrate=True,eq_limit=1e-2) ktrap = 1.2e-15

        

    
    def trPL(self,X,params,X_dimensions=[],take_log=False):
        """ Run the trPL simulations for a given list of parameters 

        Parameters
        ----------
        X : np.array
            Array of fixed parameters (like time, light intensities, etc.)) 
        params : list
            list of Fitparam objects
        X_dimensions : list, optional
            name of the fixed parameters in X, by default []

        Returns
        -------
        np.array
            Array of containing the simulation results
        """ 
        # check if X is np.array
        if not isinstance(X,np.ndarray):
            X = np.array(X)


        QE, I_PL, p_0, Gfrac = None,None,None,None
        y = []
        X_unique, X_dimensions_uni,ts = get_unique_X_and_xaxis_values(X,'t',X_dimensions) # get unique X values and their dimensions
        pnames = [p.name for p in params] # get parameter names

        
        flux_args, pump_args, pump_params, trPL_params, trPL_default_val, trPL_args_names = self.init_args_flux_pump_models() # initialize arguments for the flux density and pump models

        for idx, uni in enumerate(X_unique):
            t = ts[idx]

            # update the pump,flux_density and trPL arguments from a fixed parameter
            for key in X_dimensions_uni:
                val = uni[X_dimensions_uni.index(key)]
                if key in pnames:
                    raise ValueError(f'Parameter {key} is optimized, it should not be in the fixed parameters')

                if key in flux_args.keys() and self.pump_model != initial_carrier_density:
                    flux_args[key] = uni[X_dimensions_uni.index(key)]
                
                if key in pump_args.keys():
                    pump_args[key] = uni[X_dimensions_uni.index(key)]
                
                if key in pnames:
                    pump_params[key] = params[pnames.index(key)].value
                    warnings.warn(f'Typically Pump parameter {key} should not be optimized, it is set to {pump_params[key]}')

                if key == 'QE':
                    QE = uni[X_dimensions_uni.index(key)]
                
                if key == 'I_PL':
                    I_PL = uni[X_dimensions_uni.index(key)]
                
                if key == 'p_0':
                    p_0 = uni[X_dimensions_uni.index(key)]
                
                if key == 'Gfrac':
                    Gfrac = uni[X_dimensions_uni.index(key)]  

                if key == 'N0':
                    pump_args[key] = uni[X_dimensions_uni.index(key)]            
                else:
                    pass
            
            # update the trPL arguments from the params list
            for p in params:
                if p.name in pump_args.keys():
                    pump_args[p.name] = p.val
                if p.name in flux_args.keys():
                    flux_args[p.name] = p.val
                if p.name in trPL_args_names:
                    trPL_params[p.name] = p.val
                if p.name == 'QE':
                    QE = p.val
                if p.name == 'I_PL':
                    I_PL = p.val
                if p.name == 'p_0':
                    p_0 = p.val
                if p.name == 'Gfrac':
                    Gfrac = p.val
                if p.name == 'N0':
                    pump_args[p.name] = p.val

                
            # check if the trPL arguments are set
            if QE is None:
                QE = trPL_default_val['QE']
                warnings.warn(f'QE is not set, it is set to {QE} %')
            if I_PL is None:
                I_PL = trPL_default_val['I_PL']
                warnings.warn(f'I_PL is not set, it is set to {I_PL} ')
            if p_0 is None and self.model_name == 'Bimolecular_Trapping_Detrapping_equation':
                p_0 = trPL_default_val['p_0']
                warnings.warn(f'p_0 is not set, it is set to {p_0} cm-3')
            if Gfrac is None :
                Gfrac = 1
                pump_args['Gfrac'] = Gfrac
                # warnings.warn(f'Gfrac is not set, it is set to {Gfrac} ')
            else:
                pump_args['Gfrac'] = Gfrac
                

            
            # calculate the flux density
            if self.pump_model != initial_carrier_density:
                flux,density = self.flux_density_model(**flux_args)

                # calculate the pump
                pump_args['P'] = density # update the pump power with the density
                trPL_params['t']= t
                tmax = 0.99999*1/pump_args['fpu'] # maximum time for the pump

                # check if the time is too long to use arange, we need to do this to ensure that the pulse_width is not too small compared to the timestep
                order_of_magnitude = np.floor(np.log10(pump_args['pulse_width']))

                if tmax > 100*10**order_of_magnitude:
                    # use logspace if the time is too long
                    num_steps = (np.floor(np.log10(tmax))+1-order_of_magnitude) * 10 # at least ten steps per order of magnitude
                    tpulse = np.logspace(order_of_magnitude-1,np.log10(tmax),int(num_steps)) # time vector for the pump
                    #add 0 to the beginning of the time vector
                    tpulse = np.insert(tpulse,0,0)
                else: # use arange if the time is short enough can use linear steps
                    tpulse = np.arange(0,tmax,10**order_of_magnitude)

                trPL_params['tpulse'] = tpulse
                trPL_params['Gpulse'] = self.pump_model(tpulse,**pump_args) * QE# * Gfrac

                signal = self.trPL_model(**trPL_params)
            else:
                trPL_params['t']= t
                tmax = 0.99999*1/pump_args['fpu'] # maximum time for the pump
                tpulse = np.linspace(0,tmax,5000) # time vector for the pump
                trPL_params['tpulse'] = tpulse
                trPL_params['Gpulse'] = self.pump_model(tpulse,**pump_args) * QE #* Gfrac

                if self.model_name == 'Bimolecular_Trapping_equation': # initial electron density
                    trPL_params['ninit'] = [pump_args['N0']*Gfrac]

                elif self.model_name == 'Bimolecular_Trapping_Detrapping_equation':
                    trPL_params['ninit'] = [pump_args['N0']*Gfrac,0,pump_args['N0']*Gfrac] # initial electron, hole and exciton density

                signal = self.trPL_model(**trPL_params)

            if self.model_name == 'Bimolecular_Trapping_equation':
                signal = trPL_params['kdirect'] * I_PL * signal**2  # PL = I_PL  * kdirect * n * p, here n = p
            elif self.model_name == 'Bimolecular_Trapping_Detrapping_equation':
                n, nt , p = signal# electron density
                signal = trPL_params['kdirect'] * I_PL * n * (p + p_0) # PL = I_PL * kdirect * n * (p + p_0)
            
            y = y + list(signal) # small signal limit !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        y = np.array(y)

        if take_log:
            y = np.log10(abs(y))

        return y
                
                

    def init_args_flux_pump_models(self): 
        """ Initialize the arguments for the flux density model, pump model and trPL model 
            For more information, check the pump.py file in the dynamic_utils folder
        """

        # Reset default values when checking for model, to avoid errors

        # check model function name even when it is called inside a partial function
        model_name = callable_name(self.trPL_model)
        self.model_name = model_name
        # if self.model_name == 'Bimolecular_Trapping_equation':
        if self.model_name == 'Bimolecular_Trapping_equation':
            self.trPL_default_val = {'kdirect' : 1e-18, 'ktrap': 1e5, 'QE':0.9, 'I_PL': 1e-17} # kwargs for the trPL model  by defaults (tpulse=None, equilibrate=True,eq_limit=1e-2)
        # elif self.trPL_model == Bimolecular_Trapping_Detrapping_equation:
        elif self.model_name == 'Bimolecular_Trapping_Detrapping_equation':
            self.trPL_default_val = {'kdirect' : 26e-17, 'ktrap': 1.2e-15, 'kdetrap': 80e-17,'Bulk_tr':6e18,'p_0':65e18, 'QE':0.9, 'I_PL': 1e-17} # kwargs for the trPL model  by defaults (tpulse=None, equilibrate=True,eq_limit=1e-2) ktrap = 1.2e-15
        else:
            self.trPL_default_val = {}


        # Initialize the dictionary with the default values and make deep copy
        trPL_default_val = deepcopy(self.trPL_default_val)
        pump_params = deepcopy(self.pump_params)


        # get arguments for get_flux_density
        if self.pump_model != initial_carrier_density: # if the pump model is not the initial carrier density, then the flux density model is used
            if self.flux_density_model == get_flux_density:
                # P,wvl,fpu,A,alpha
                flux_args = {'wvl':self.pump_params['wvl'], 'fpu':self.pump_params['fpu'], 'A':self.pump_params['A'], 'alpha':self.pump_params['alpha'],'P':self.pump_params['P']}
        else:
            flux_args = {}    
        
        # get arguments for pump_model
        # P is ommited because it is calculated in get_flux_density
        if self.pump_model == square_pump:
            pump_args = {'fpu':self.pump_params['fpu'],'pulse_width':self.pump_params['pulse_width'], 't0':self.pump_params['t0'], 'background':self.pump_params['background']}
        elif self.pump_model == gaussian_pump:
            pump_args = {'fpu':self.pump_params['fpu'],'pulse_width':self.pump_params['pulse_width'], 't0':self.pump_params['t0'], 'background':self.pump_params['background']}
        elif self.pump_model == pump_from_file:
            pump_args = {'filename':self.pump_params['filename'], 'P':self.pump_params['P'], 'background':self.pump_params['background'], 'sep':self.pump_params['sep']}
        elif self.pump_model == initial_carrier_density:
            pump_args = {'fpu':self.pump_params['fpu'],'background':self.pump_params['background']}
            if 'N0' in self.pump_params.keys():
                pump_args['N0'] = self.pump_params['N0']
        else:
            raise ValueError(f'pump model {self.pump_model} not implemented')

        # get argument for trPL_model
        if self.model_name == 'Bimolecular_Trapping_equation':
            trPL_args_names = ['ktrap','kdirect'] # name of the physical parameters in the Bimolecular_Trapping_equation
            for i in trPL_args_names:
                if i not in self.trPL_default_val.keys():
                    trPL_default_val[i] = None
            trPL_params = {'ktrap':self.trPL_default_val['ktrap'],'kdirect':self.trPL_default_val['kdirect']} # default values of the physical parameters in the Bimolecular_Trapping_equation
        elif self.model_name  == 'Bimolecular_Trapping_Detrapping_equation':
            trPL_args_names = ['ktrap', 'kdirect', 'kdetrap', 'Bulk_tr', 'p_0']
            for i in trPL_args_names:
                if i not in self.trPL_default_val.keys():
                    trPL_default_val[i] = None
            trPL_params = {'ktrap':self.trPL_default_val['ktrap'],'kdirect':self.trPL_default_val['kdirect'], 'kdetrap':self.trPL_default_val['kdetrap'], 'Bulk_tr':self.trPL_default_val['Bulk_tr'], 'p_0':self.trPL_default_val['p_0']}
        else:
            raise ValueError(f'trPL model {self.model_name} not implemented')
        
        return flux_args, pump_args, pump_params, trPL_params, trPL_default_val, trPL_args_names