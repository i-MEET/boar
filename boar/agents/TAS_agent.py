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

class TAS_agent(Agent):
    """ Agent to run Transient Absorption Spectroscopy (TAS) simulation based on rate equations to be used with BOAR MultiObjectiveOptimizer
    
    Parameters
    ----------
    TAS_model : function, optional
        TAS model to be used, by default Bimolecular_Trapping_equation
        To see the available models, check the rate_eq.py file in the dynamic_utils folder

    pump_model : function, optional
        pump model to be used, by default square_pump
        To see the available models, check the pump.py file in the dynamic_utils folder

    pump_params : dict, optional
        dictionary of pump parameters, by default {'P':0.0039, 'wvl':850, 'fpu':10000, 'A':0.3*0.3*1e-4, 'alpha':1e-5*1e-2, 'pulse_width':0.2*(1/10000), 't0':0, 'background':0}

            including:
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
    def __init__(self,TAS_model = Bimolecular_Trapping_equation,pump_model = square_pump,pump_params = {'P':0.0039, 'wvl':850, 'fpu':10000, 'A':0.3*0.3*1e-4, 'alpha':1e-5*1e-2, 'pulse_width':0.2*(1/10000), 't0':0, 'background':0},flux_density_model = get_flux_density) -> None:
        super().__init__()
        # self.flux_density_model = flux_density_model
        # self.TAS_model = TAS_model
        # self.pump_model = pump_model
        # self.pump_params= pump_params
        # if self.TAS_model == Bimolecular_Trapping_equation:
        #     self.TAS_default_val = {'kdirect' : 1e-18, 'ktrap': 1e5, 'QE':0.9, 'crossec': 1e-17, 'thickness': 100e-9 } # kwargs for the TAS model  by defaults (tpulse=None, equilibrate=True,eq_limit=1e-2)
        # elif self.TAS_model == Bimolecular_Trapping_Detrapping_equation:
        #     self.TAS_default_val = {'kdirect' : 26e-17, 'ktrap': 1.2e-15, 'kdetrap': 80e-17,'Bulk_tr':6e18,'p_0':65e18,'QE':0.9, 'crossec': 1e-17, 'thickness': 100e-9 } # kwargs for the TAS model  by defaults (tpulse=None, equilibrate=True,eq_limit=1e-2) ktrap = 1.2e-15
        
        self.flux_density_model = flux_density_model
        self.TAS_model = TAS_model
        self.pump_model = pump_model
        self.pump_params= pump_params


        #  check model function name even when it is called inside a partial function
        model_name = callable_name(self.TAS_model)
        if model_name == 'Bimolecular_Trapping_equation':
            self.TAS_default_val = {'kdirect' : 1e-18, 'ktrap': 1e5, 'QE':0.9, 'crossec': 1e-17, 'thickness': 100e-9 } # kwargs for the TAS model  by defaults (tpulse=None, equilibrate=True,eq_limit=1e-2)
        elif model_name == 'Bimolecular_Trapping_Detrapping_equation':
            self.TAS_default_val = {'kdirect' : 26e-17, 'ktrap': 1.2e-15, 'kdetrap': 80e-17,'Bulk_tr':6e18,'p_0':65e18,'QE':0.9, 'crossec': 1e-17, 'thickness': 100e-9 } # kwargs for the TAS model  by defaults (tpulse=None, equilibrate=True,eq_limit=1e-2) ktrap = 1.2e-15
        else:
            self.TAS_default_val = {}

        self.model_name = model_name

        

    

    def TAS(self,X,params,X_dimensions=[],take_log = False):
        """ Run the TAS simulations for a given list of parameters 

        Parameters
        ----------
        X : np.array
            Array of fixed parameters (like time, light intensities, etc.)) 
        params : list
            list of Fitparam objects
        X_dimensions : list, optional
            name of the fixed parameters in X, by default []
        take_log : bool, optional
            if True, the simulation results are taken in log10, by default False

        Returns
        -------
        np.array
            Array of containing the simulation results
        """ 
        # check if X is np.array
        if not isinstance(X,np.ndarray):
            X = np.array(X)


        QE, crossec, thickness, Gfrac = None,None,None,None
        y = []
        X_unique, X_dimensions_uni,ts = get_unique_X_and_xaxis_values(X,'t',X_dimensions) # get unique X values and their dimensions
        pnames = [p.name for p in params] # get parameter names

        thickness = self.pump_params['thickness']
        flux_args, pump_args, pump_params, TAS_params, TAS_default_val, TAS_args_names = self.init_args_flux_pump_models() # initialize arguments for the flux density and pump models

     
        for idx, uni in enumerate(X_unique):
            t = ts[idx]

            # update the pump,flux_density and TAS arguments from a fixed parameter
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
                
                if key == 'crossec':
                    crossec = uni[X_dimensions_uni.index(key)]
                
                if key == 'thickness':
                    thickness = uni[X_dimensions_uni.index(key)]
                
                if key == 'Gfrac':
                    Gfrac = uni[X_dimensions_uni.index(key)]

                if key == 'N0':
                    pump_args[key] = uni[X_dimensions_uni.index(key)] 

                else:
                    pass
            
            # update the TAS arguments from the params list
            for p in params:
                if p.name in pump_args.keys():
                    pump_args[p.name] = p.val
                if p.name in flux_args.keys():
                    flux_args[p.name] = p.val
                if p.name in TAS_args_names:
                    TAS_params[p.name] = p.val
                if p.name == 'QE':
                    QE = p.val
                if p.name == 'crossec':
                    crossec = p.val
                if p.name == 'thickness':
                    thickness = p.val
                if p.name == 'Gfrac':
                    Gfrac = p.val
                if p.name == 'N0':
                    pump_args['N0'] = p.val
                
            # check if the TAS arguments are set
            if QE is None:
                QE = TAS_default_val['QE']
                warnings.warn(f'QE is not set, it is set to {QE} %')
            if crossec is None:
                crossec = TAS_default_val['crossec']
                warnings.warn(f'crossec is not set, it is set to {crossec} m^2')
            if thickness is None:
                # check is the thickness is set in the pump model
                if 'thickness' in pump_args.keys():
                    thickness = pump_args['thickness']
                else:
                    thickness = TAS_default_val['thickness']
                    warnings.warn(f'thickness is not set, it is set to {thickness} m')
            if Gfrac is None :
                Gfrac = 1
                # warnings.warn(f'Gfrac is not set, it is set to {Gfrac} ')
                pump_args['Gfrac'] = Gfrac
            else:
                pump_args['Gfrac'] = Gfrac

                            
            
            # # calculate the flux density
            # flux,density = self.flux_density_model(**flux_args)
            # # calculate the pump
            # pump_args['P'] = density # update the pump power with the density
            # TAS_params['t']= t
            # tmax = 0.99999*1/pump_args['fpu'] # maximum time for the pump
            # tpulse = np.linspace(0,tmax,1000) # time vector for the pump
            # TAS_params['tpulse'] = tpulse
            # TAS_params['Gpulse'] = self.pump_model(tpulse,**pump_args) * QE #* Gfrac # update the pump with the density
            
            # signal = self.TAS_model(**TAS_params)

            
            if self.pump_model == initial_carrier_density:
                TAS_params['t']= t
                tmax = 0.99999*1/pump_args['fpu'] # maximum time for the pump
                tpulse = np.linspace(0,tmax,5000) # time vector for the pump
                TAS_params['tpulse'] = tpulse
                TAS_params['Gpulse'] = self.pump_model(tpulse,**pump_args) * QE #* Gfrac

                if self.model_name == 'Bimolecular_Trapping_equation': # initial electron density
                    TAS_params['ninit'] = [pump_args['N0']*Gfrac]

                elif self.model_name == 'Bimolecular_Trapping_Detrapping_equation':
                    TAS_params['ninit'] = [pump_args['N0']*Gfrac,0,pump_args['N0']*Gfrac] # initial electron, hole and exciton density

                signal = self.TAS_model(**TAS_params)
            
            elif self.pump_model == square_pump:
                # calculate the flux density
                flux,density = self.flux_density_model(**flux_args)
                # calculate the pump
                pump_args['P'] = density # update the pump power with the density
                TAS_params['t']= t
                tmax = 0.99999*1/pump_args['fpu'] # maximum time for the pump
                tpulse = np.linspace(0,tmax,1000) # time vector for the pump
                TAS_params['tpulse'] = tpulse
                TAS_params['Gpulse'] = self.pump_model(tpulse,**pump_args) * QE #* Gfrac # update the pump with the density
                
                signal = self.TAS_model(**TAS_params)
            elif self.pump_model == gaussian_pump:
                flux,density = self.flux_density_model(**flux_args)

                # calculate the pump
                pump_args['P'] = density # update the pump power with the density
                TAS_params['t']= t
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

                TAS_params['tpulse'] = tpulse
                TAS_params['Gpulse'] = self.pump_model(tpulse,**pump_args) * QE# * Gfrac
                # plt.plot(tpulse,TAS_params['Gpulse'])

                signal = self.TAS_model(**TAS_params)
            else:
                raise ValueError('Pump model not recognized please use one of the following: initial_carrier_density, square_pump, gaussian_pump')

            signal-=np.mean(signal)# simulate AC coupling 
            
            y = y + list(signal) # small signal limit !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        y = np.array(y)
        y = np.exp(-crossec * thickness * y) - 1  # lambert-Beer !!!!!!!!!!!!!! milli-OD!!! 

        if take_log:
            y = np.log10(abs(y))

        return y
                
                

    def init_args_flux_pump_models(self): 
        """ Initialize the arguments for the flux density model, pump model and TAS model 
            For more information, check the pump.py file in the dynamic_utils folder
        """

        # Reset default values when checking for model, to avoid errors

        # check model function name even when it is called inside a partial function
        model_name = callable_name(self.TAS_model)
        self.model_name = model_name
        if self.model_name == 'Bimolecular_Trapping_equation':
            self.TAS_default_val = {'kdirect' : 1e-18, 'ktrap': 1e5, 'QE':0.9, 'crossec': 1e-17, 'thickness': 100e-9 } # kwargs for the TAS model  by defaults (tpulse=None, equilibrate=True,eq_limit=1e-2)
        elif self.model_name == 'Bimolecular_Trapping_Detrapping_equation':
            self.TAS_default_val = {'kdirect' : 26e-17, 'ktrap': 1.2e-15, 'kdetrap': 80e-17,'Bulk_tr':6e18,'p_0':65e18,'QE':0.9, 'crossec': 1e-17, 'thickness': 100e-9 } # kwargs for the TAS model  by defaults (tpulse=None, equilibrate=True,eq_limit=1e-2) ktrap = 1.2e-15
        else:
            self.TAS_default_val = {}

        

        # Initialize the dictionary with the default values and make deep copy
        TAS_default_val = deepcopy(self.TAS_default_val)
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

        # get argument for TAS_model
        if self.model_name == 'Bimolecular_Trapping_equation':
            TAS_args_names = ['ktrap','kdirect'] # name of the physical parameters in the Bimolecular_Trapping_equation
            for i in TAS_args_names:
                if i not in self.TAS_default_val.keys():
                    TAS_default_val[i] = None
            TAS_params = {'ktrap':self.TAS_default_val['ktrap'],'kdirect':self.TAS_default_val['kdirect']} # default values of the physical parameters in the Bimolecular_Trapping_equation
        elif self.model_name == 'Bimolecular_Trapping_Detrapping_equation':
            TAS_args_names = ['ktrap', 'kdirect', 'kdetrap', 'Bulk_tr', 'p_0']
            for i in TAS_args_names:
                if i not in self.TAS_default_val.keys():
                    TAS_default_val[i] = None
            TAS_params = {'ktrap':self.TAS_default_val['ktrap'],'kdirect':self.TAS_default_val['kdirect'], 'kdetrap':self.TAS_default_val['kdetrap'], 'Bulk_tr':self.TAS_default_val['Bulk_tr'], 'p_0':self.TAS_default_val['p_0']}
        else:
            raise ValueError(f'TAS model {self.model_name} not implemented')
        
        return flux_args, pump_args, pump_params, TAS_params, TAS_default_val, TAS_args_names

    def get_pseudo_JV(self,params, powers, Eg, Nc, T = 293):
        """Calculate the pseudo JV curve from the stationary charge concentration
        using the TAS model.
        See: https://doi.org/10.1002/adma.202000080
        and https://doi.org/10.1002/solr.202000649

        QFLS = Eg - kT * ln(Nc*Nv/n*p) 
        
        here we assume Nc = Nv

        Parameters
        ----------
        params : list of FitParam objects
            list with the parameters that were fitted
        powers : 1D sequence of floats
            list of the pump powers
        Eg : float
            bandgap of the material, if Eg is not params then Eg here is used
        Nc : float
            effective density of states [m^-3] if Nc is not params then Nc here is used
        T : float, optional
            temperature in K, by default 293

        Returns
        -------
        Xqfls : 2D array

        """        

        # We chose a very slow repetition rate (fpu) to get a good approximation of the stationary charge concentration
        fpu = 100 # may be too fast for super small light intensties
        pulse_width  = 0.99 * 1/fpu # duty cycle of the pump pulse
        t = np.linspace(0,1/fpu,101) # time axis
        Xqfls = []
        X_dimensions_qfls = ['t','P','fpu','pulse_width']
        for P in powers: # iterate over all powers
            for tt in t:
                Xqfls.append([tt,P,fpu,pulse_width])
        Xqfls = np.array(Xqfls)

        rm = []
        currents = []

        # Run TAS
        QE, crossec, thickness = None,None,None
        y = []
        X_unique, X_dimensions_uni,ts = get_unique_X_and_xaxis_values(Xqfls,'t',X_dimensions_qfls) # get unique X values and their dimensions
        pnames = [p.name for p in params] # get parameter names

        
        flux_args, pump_args, pump_params, TAS_params, TAS_default_val, TAS_args_names = self.init_args_flux_pump_models() # initialize arguments for the flux density and pump models
        pump_args['pulse_width'] = pulse_width
        pump_args['fpu'] = fpu

        for idx, uni in enumerate(X_unique):
            t = ts[idx]

            # update the pump,flux_density and TAS arguments from a fixed parameter
            for key in X_dimensions_uni:
                val = uni[X_dimensions_uni.index(key)]
                if key in pnames:
                    raise ValueError(f'Parameter {key} is optimized, it should not be in the fixed parameters')
                    
                if key in flux_args.keys():
                    flux_args[key] = uni[X_dimensions_uni.index(key)]
                
                if key in pump_args.keys():
                    pump_args[key] = uni[X_dimensions_uni.index(key)]
                
                if key in pnames:
                    pump_params[key] = params[pnames.index(key)].value
                    warnings.warn(f'Typically Pump parameter {key} should not be optimized, it is set to {pump_params[key]}')
                else:
                    pass
            
            # update the TAS arguments from the params list
            for p in params:
                if p.name in pump_args.keys():
                    pump_args[p.name] = p.val
                if p.name in flux_args.keys():
                    flux_args[p.name] = p.val
                if p.name in TAS_args_names:
                    TAS_params[p.name] = p.val
                if p.name == 'QE':
                    QE = p.val
                if p.name == 'crossec':
                    crossec = p.val
                if p.name == 'thickness':
                    thickness = p.val
                
            # check if the TAS arguments are set
            if QE is None:
                QE = TAS_default_val['QE']
                warnings.warn(f'QE is not set, it is set to {QE} %')
            if crossec is None:
                crossec = TAS_default_val['crossec']
                warnings.warn(f'crossec is not set, it is set to {crossec} m^2')
            if thickness is None:
                thickness = TAS_default_val['thickness']
                warnings.warn(f'thickness is not set, it is set to {thickness} m')
                            
            
            # calculate the flux density
            flux,density = self.flux_density_model(**flux_args)

            # calculate the pump
            pump_args['P'] = density # update the pump power with the density
            TAS_params['t']= t
            tmax = 0.99999*1/pump_args['fpu'] # maximum time for the pump
            tpulse = np.linspace(0,tmax,5000) # time vector for the pump
            TAS_params['tpulse'] = tpulse
            TAS_params['Gpulse'] = self.pump_model(tpulse,**pump_args) * QE 

            signal = self.TAS_model(**TAS_params)
            rm.append(max(signal)) # stationary charge concentration
            currents.append(flux*fpu*q*QE) # in A m-2, The current is the flux density times the repetition rate times the elementary charge times the quantum efficiency
        
        # Update the necessary parameter values for the calculation
        if 'Eg' in pnames:
            Eg = params[pnames.index('Eg')].val
        if 'Nc' in pnames:
            Nc = params[pnames.index('Nc')].val
        
        rm = np.array(rm)
        currents = np.array(currents)
        # calculate the QFLS
        QFLS = Eg + kb*T * np.log(rm*rm/Nc**2) # QFLS in eV see https://doi.org/10.1016/j.xcrp.2021.100346

        return currents, QFLS





        

                
            


                    
                        


                    

