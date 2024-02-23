######################################################################
#################### BOAR Optimizer Class ############################
######################################################################
# Authors: 
# Larry Lueer (https://github.com/larryluer)
# Vincent M. Le Corre (https://github.com/VMLC-PV)
# (c) i-MEET, University of Erlangen-Nuremberg, 2021-2022-2023 


# Import libraries
import os,copy,warnings
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from copy import deepcopy
from sklearn.metrics import mean_squared_error,mean_squared_log_error,mean_absolute_error,mean_absolute_percentage_error
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import curve_fit,basinhopping,dual_annealing
from functools import partial
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
# Import boar libraries
from boar.core.funcs import sci_notation
from boar.core.funcs import get_unique_X


class BoarOptimizer():
    """ Provides a default class for the different optimizers in BOAR. \\
    This class is not intended to be used directly, but rather to be inherited by the different optimizer classes. \\
    It provides the basic functionality for the optimizer classes, such as the different objective functions, \\
    the functions to handle the Fitparam() objects, and the functions to handle the different plotting options. \\

    
    """    
    # a class for multi-objective optimization
    def __init__(self) -> None:
        pass

    
    def params_r(self, params):
        """ Prepare starting guess and bounds for optimizer  
        Considering settings of Fitparameters:  

            optim_type: 
                if 'linear': abstract the order of magnitude (number between 1 and 10)  
                if 'log': bring to decadic logarithm (discarding the negative sign, if any!)  
            lim_type:   if 'absolute', respect user settings in Fitparam.lims  
                if 'relative', respect user settings in Fitparam.relRange: 

                    if Fitparam.range_type=='linear':  
                        interpret Fitparam.relRange as factor  
                    if Fitparam.range_type=='log':  
                        interpret Fitparam.relRange as order of magnitude  
            relRange:   if zero, Fitparam is not included into starting guess and bounds  
                        (but still available to the objective function)  
        

        Parameters
        ----------
        params : list
            list of Fitparam() objects

        Returns
        -------
        set of lists
            set of lists with:  p0 = initial guesses\\
                                lb = lower bounds\\
                                ub = upper bounds\\       
        """

        lb = [] # initialize vector for lower bounds
        ub = [] # initialize vector for upper bounds
        p0 = [] # initialize vector for initial guess
        for par in params: # walk across all parameter lists
            if par.val_type == 'float':
                if par.relRange != 0:
                    if par.optim_type=='linear':
                        if par.rescale == True:
                            if par.p0m == None:
                                if par.startVal == 0:
                                    if par.lims[0] != 0:
                                        p0m = 10**(np.floor(np.log10(np.abs(par.lims[0])))) # the order of magnitude of the parameters
                                        par.p0m = p0m # save it in the Fitparam so it does not need to be calculated again
                                    elif par.lims[1] != 0:
                                        p0m = 10**(np.floor(np.log10(np.abs(par.lims[1])))) # the order of magnitude of the parameters
                                        par.p0m = p0m # save it in the Fitparam so it does not need to be calculated again
                                    else:
                                        par.p0m = 1
                                else:
                                    p0m = 10**(np.floor(np.log10(np.abs(par.startVal)))) # the order of magnitude of the parameters
                                    par.p0m = p0m # save it in the Fitparam so it does not need to be calculated again
                            else:
                                p0m = par.p0m
                            p0.append(par.startVal/p0m) # optimizer can't handle very large besides very small numbers
                                                        # therefore, bring values to same order of magnitude
                            if par.lim_type == 'absolute':
                                lb0 = par.lims[0]
                                ub0 = par.lims[1]
                            elif par.lim_type == 'relative':
                                if par.range_type == 'linear':
                                    lb0 = (par.startVal - par.relRange * abs(par.startVal))   # this is linear version
                                    ub0 = (par.startVal + par.relRange * abs(par.startVal))   # this is linear version    
                                    par.lims[0] = lb0
                                    par.lims[1] = ub0                            
                                elif par.range_type == 'log':
                                    lb0 = par.startVal*10**(-par.relRange)  # this is logarithmic version
                                    ub0 = par.startVal*10**(par.relRange)  # this is logarithmic version
                                    par.lims[0] = lb0
                                    par.lims[1] = ub0
                            lb.append(lb0/p0m)
                            ub.append(ub0/p0m)
                        else:
                            par.p0m = 1.0
                            p0.append(par.startVal)
                            if par.lim_type == 'absolute':
                                lb.append(par.lims[0])
                                ub.append(par.lims[1])
                            elif par.lim_type == 'relative':
                                if par.range_type == 'linear':
                                    lb.append(par.startVal - par.relRange * abs(par.startVal))
                                    ub.append(par.startVal + par.relRange * abs(par.startVal))
                                elif par.range_type == 'log':
                                    lb.append(par.startVal*10**(-par.relRange))
                                    ub.append(par.startVal*10**(par.relRange))


                    elif par.optim_type=='log':
                        if par.rescale == True:
                            if par.p0m == None:
                                # optimizer won't have problems with logarithmic quantities
                                if par.startVal == 0:
                                    p0.append(-10)
                                    par.startVal = 1e-10
                                else:
                                    if par.startVal < 0:
                                        print('WARNING: negative start value for parameter ' + par.name + ' will be converted to positive value for optimization')
                                    p0.append(np.log10(abs(par.startVal)))
                                if par.lim_type == 'absolute':
                                    lb.append(np.log10(par.lims[0]))
                                    ub.append(np.log10(par.lims[1]))
                                elif par.lim_type == 'relative':
                                    if par.range_type == 'linear':
                                        lb.append(np.log10(par.startVal - par.relRange * abs(par.startVal)))   # this is linear version
                                        ub.append(np.log10(par.startVal + par.relRange * abs(par.startVal)))    # this is linear version                                
                                    elif par.range_type == 'log':
                                        lb.append(np.log10(par.startVal*10**(-par.relRange)))  # this is logarithmic version
                                        ub.append(np.log10(par.startVal*10**(par.relRange)))  # this is logarithmic version
                                
                            else:
                                if par.startVal == 0:
                                    p0.append(np.log10(1e-10/par.p0m))
                                    par.startVal = 1e-10/par.p0m
                                else:
                                    if par.startVal < 0:
                                        print('WARNING: negative start value for parameter ' + par.name + ' will be converted to positive value for optimization')
                                    p0.append(np.log10(abs(par.startVal/par.p0m)))
                                if par.lim_type == 'absolute':
                                    lb.append(np.log10(par.lims[0]/par.p0m))
                                    ub.append(np.log10(par.lims[1]/par.p0m))
                                elif par.lim_type == 'relative':
                                    if par.range_type == 'linear':
                                        lb.append(np.log10((par.startVal - par.relRange * abs(par.startVal))/par.p0m))
                                        ub.append(np.log10((par.startVal + par.relRange * abs(par.startVal))/par.p0m))
                                    elif par.range_type == 'log':
                                        lb.append(np.log10(par.startVal*10**(-par.relRange)/par.p0m))
                                        ub.append(np.log10(par.startVal*10**(par.relRange)/par.p0m))
                        else:
                            if par.p0m == None:
                                # optimizer won't have problems with logarithmic quantities
                                if par.startVal == 0:
                                    p0.append(1e-10)
                                    par.startVal = 1e-10
                                else:
                                    if par.startVal < 0:
                                        print('WARNING: negative start value for parameter ' + par.name + ' will be converted to positive value for optimization')
                                    p0.append(par.startVal)
                                if par.lim_type == 'absolute':
                                    lb.append(par.lims[0])
                                    ub.append(par.lims[1])
                                elif par.lim_type == 'relative':
                                    if par.range_type == 'linear':
                                        lb.append(par.startVal - par.relRange * abs(par.startVal))
                                        ub.append(par.startVal + par.relRange * abs(par.startVal))
                                    elif par.range_type == 'log':
                                        lb.append(par.startVal*10**(-par.relRange))
                                        ub.append(par.startVal*10**(par.relRange))
                                
                            else:
                                if par.startVal == 0:
                                    p0.append(1e-10/par.p0m)
                                    par.startVal = 1e-10/par.p0m
                                else:
                                    if par.startVal < 0:
                                        print('WARNING: negative start value for parameter ' + par.name + ' will be converted to positive value for optimization')
                                    p0.append(par.startVal/par.p0m)
                                if par.lim_type == 'absolute':
                                    lb.append(par.lims[0]/par.p0m)
                                    ub.append(par.lims[1]/par.p0m)
                                elif par.lim_type == 'relative':
                                    if par.range_type == 'linear':
                                        lb.append((par.startVal - par.relRange * abs(par.startVal))/par.p0m)
                                        ub.append((par.startVal + par.relRange * abs(par.startVal))/par.p0m)
                                    elif par.range_type == 'log':
                                        lb.append(par.startVal*10**(-par.relRange)/par.p0m)
                                        ub.append(par.startVal*10**(par.relRange)/par.p0m)
                else:
                    par.p0m = 1
            elif par.val_type == 'int':
                if par.relRange != 0:
                    if par.stepsize == None or par.rescale == False:
                        par.p0m = 1
                    else:
                        par.p0m = par.stepsize
                else:
                    par.p0m = 1

                if par.relRange != 0:
                    lb.append(round(par.lims[0]/par.p0m))
                    ub.append(round(par.lims[1]/par.p0m))
                    p0.append(par.startVal)
            else:
                par.p0m = 1

        #print('params_r returns p0, lb, ub')
        #print(p0, lb, ub)
        return p0, lb, ub

    def params_w(self, x, params, std = [], which='val'):
        """Method to interact with Fitparam objects\\
        Used by all Obj_funcs to write desired parameter from optimizer 
        so the model can have the parameters in the physically correct units
        The fitparams objects are in a nested list at self.include_params

        Parameters
        ----------
        x : 1D-sequence of floats
            fit parameters as requested by optimizer
        params : list
            list of Fitparam() objects
        std : list, optional
            Contains the 95% confidence interval of the parameters, by default []
        which : str, optional
            'val': x => Fitparam.val\\
            'startVal': x=> Fitparam.startVal\\
            defaults to Fitparam.val (which is used by the obj_funcs\\
            to pass to the model function), by default 'val'

        Returns
        -------

        """  

        ip = 0 # initialize param counter
        
        if len(std)==0: # if no standard deviation is given, set lims to x
            for i in x:
                std.append((i,i))
            # std = x

        for par in params:
            if par.val_type == 'float':
                if par.relRange != 0:
                    if par.optim_type == 'linear':
                        if which=='val':
                            par.val = x[ip] * par.p0m # bring back to correct order of magnitude, if necessary
                        elif which=='startVal':
                            par.startVal = x[ip] * par.p0m
                        par.std = std[ip] #* par.p0m

                    elif par.optim_type == 'log':
                        if par.rescale == True:
                            if par.p0m == None:
                                if which=='val':
                                    par.val = 10**(x[ip])
                                elif which=='startVal':
                                    par.startVal = 10**(x[ip])
                            else:
                                if which=='val':
                                    par.val = 10**(x[ip])*par.p0m
                                elif which=='startVal':
                                    par.startVal = 10**(x[ip])*par.p0m
                        else:
                            if par.p0m == None:
                                if which=='val':
                                    par.val = x[ip]
                                elif which=='startVal':
                                    par.startVal = x[ip]
                            else:
                                if which=='val':
                                    par.val = x[ip]*par.p0m
                                elif which=='startVal':
                                    par.startVal = x[ip]*par.p0m

                        par.std = std[ip] # !!!!!! This is logarithmic while par.val is delogarithmized!!!!!
                    else: 
                        raise ValueError('ERROR. ',par.name,' optim_type needs to be ''linear'' or ''log'' not ',par.optim_type,'.')      
                    ip += 1
            elif par.val_type == 'int':
                if par.relRange != 0:
                    if which=='val':
                        par.val = int(x[ip] * par.p0m)
                    elif which=='startVal':
                        par.startVal = int(x[ip] * par.p0m)
                    par.std = std[ip]
                    ip += 1
            else:
                if which=='val':
                    par.val = x[ip]
                elif which=='startVal':
                    par.startVal = par.startVal
                par.std = (0,0)
                ip += 1

    def obj_func_metric(self,target,yf,obj_type='MSE'):
        """ Method to calculate the objective function value\\
        Different objective functions can be used, see below


        Parameters
        ----------
        target : target object
            target object with data and weight
        yf : array
            model output
        obj_type : str, optional
            objective function type, can be ['MSE', 'RMSE', 'MSLE','nRMSE','MAE','MAPE','larry','nRMSE_VLC'], by default 'MSE'
            'MSE': mean squared error
            'RMSE': root mean squared error
            'MSLE': mean squared log error
            'nRMSE': normalized root mean squared error (RMSE/(max(y,yf)-min(y,yf)))
            'MAE': mean absolute error
            'MAPE': mean absolute percentage error
            'RAE': relative absolute error sum(abs(yf-y))/sum(mean(y)-y)
            'larry': mean squared error legacy version
            'nRMSE_VLC': normalized root mean squared error (RMSE/(max(y,yf)-min(y,yf))) for each experiment separately and then averaged over all experiments
            

        Returns
        -------
        float
            objective function value

        Raises
        ------
        ValueError
            if obj_type is not in ['MSE', 'RMSE', 'MSLE','nRMSE','MAE','larry','nRMSE_VLC']

        """        
        if type(yf) == list: # if yf is a list convert to array
            yf = np.asarray(yf)

        y = target['data']['y']
        weight = target['weight']
        # set weight rigth for the sklearn functions
        if type(weight) == int: # if weight is a number
            if weight == 1: # if no weight is given
                weight = None

        if obj_type == 'MSE': # mean squared error
            return mean_squared_error(yf,y,sample_weight=weight)
        
        elif obj_type == 'RMSE':
            return mean_squared_error(yf,y,sample_weight=weight,squared=False)
        
        elif obj_type == 'MSLE': # mean squared log error
            count = 0
            for i in range(len(yf)):
                if yf[i] * y[i] < 0: # if the sign of the model and data are different
                    count += 1
            if count/len(yf) > 0.8:
                warnings.warn('WARNING: more than 80$%$ of the data has different sign in model and data. The calculation will continue but we recommand  using a different objective function.', UserWarning)
            return mean_squared_log_error(abs(yf),abs(y),sample_weight=weight)
        
        elif obj_type == 'nRMSE': # normalized root mean squared error
            maxi = max(np.max(y),np.max(yf))
            mini = min(np.min(y),np.min(yf))
            return mean_squared_error(yf,y,sample_weight=weight,squared=False)/(maxi-mini)

        elif obj_type == 'MAE': # mean absolute error
            return mean_absolute_error(yf,y,sample_weight=weight)

        elif obj_type == 'MAPE': # mean absolute percentage error
            return mean_absolute_percentage_error(yf,y,sample_weight=weight)

        elif obj_type == 'RAE': # relative absolute error
            numerator = np.sum(np.abs(yf-y)*weight)
            denominator = np.sum(np.abs(np.mean(y)-y)*weight)
            return numerator/denominator
        
        elif obj_type == 'larry': # legacy 
            if weight is None: # if no weight is given
                weight = 1
            return np.mean(((yf-y)*weight)**2)
        
        elif obj_type == 'nRMSE_VLC': # normalized root mean squared error
            # split the data for unique values of X
            X = target['data']['X']
            X_dimensions = target['data']['X_dimensions']
            if 'xaxis' in X_dimensions:
                xaxis = X_dimensions['xaxis']
            else:
                xaxis = X_dimensions[0] # if no xaxis is given, take the first dimension

            X_unique,X_dimensions_uni = get_unique_X(X,xaxis,X_dimensions)

            # calculate the nRMSE for each unique value of X
            nRMSE = []
            for i in range(len(X_unique)):
                X_dum = deepcopy(X)
                #remove xaxis column from X
                idx_xaxis = X_dimensions.index(xaxis)
                X_dum = np.delete(X_dum,idx_xaxis,1) #remove xaxis column from X

                # find index of all of the rows where X_dum == X_unique[i]
                idx = np.where(np.all(X_dum==X_unique[i],axis=1))[0]

                if type(weight) == int or weight is None:
                    weight_dum = None
                else:
                    weight_dum = weight[idx]
                maxi = max(np.max(y[idx]),np.max(yf[idx]))
                mini = min(np.min(y[idx]),np.min(yf[idx]))
                nRMSE.append(mean_squared_error(yf[idx],y[idx],sample_weight=weight_dum,squared=False)/(maxi-mini))
            return np.mean(nRMSE)
        elif obj_type == 'hausdorff':

            # split the data for unique values of X
            X = target['data']['X']
            X_dimensions = target['data']['X_dimensions']
            if 'xaxis' in X_dimensions:
                xaxis = X_dimensions['xaxis']
            else:
                xaxis = X_dimensions[0] # if no xaxis is given, take the first dimension

            X_unique,X_dimensions_uni = get_unique_X(X,xaxis,X_dimensions)

            # calculate the nRMSE for each unique value of X
            dist = []
            for i in range(len(X_unique)):
                X_dum = deepcopy(X)
                #remove xaxis column from X
                idx_xaxis = X_dimensions.index(xaxis)
                X_dum = np.delete(X_dum,idx_xaxis,1) #remove xaxis column from X

                # find index of all of the rows where X_dum == X_unique[i]
                idx = np.where(np.all(X_dum==X_unique[i],axis=1))[0]

                X_ = X[idx,idx_xaxis]
                y_ = y[idx]
                yf_ = yf[idx]

                dist.append(directed_hausdorff(np.concatenate((X_.reshape(-1,1),y_.reshape(-1,1)),axis=1).T, np.concatenate((X_.reshape(-1,1),yf_.reshape(-1,1)),axis=1).T)[0])
            
            return np.sum(dist)
        elif obj_type == 'identity':
            return yf
        elif obj_type == 'diff':

            if type(yf) == float or type(yf) == int or type(yf) == np.float64 or type(yf) == np.int64:
                return  abs(yf - y[0])
            else:
                raise ValueError('yf is not a float or int')

        else:
            warnings.warn('WARNING: obj_func type not recognized. Using MSE instead.')
            return mean_squared_error(yf,y,sample_weight=weight)     

    def lossfunc(self,z0,loss,threshold=1000):
        """Define the different loss functions that can be used to calculate the 
        objective function value.

        Parameters
        ----------
        z0 : 1D-array
            data array of size (n,) with the mean squared error values
        loss : str
            type of loss function to use, can be ['linear','soft_l1','huber','cauchy','arctan']
        threshold : int, optional
            critical value above which loss sets in, by default 1000. 
            Wisely select so that the loss function affects only outliers.
            If threshold is set too low, then c<z0 even fon non-outliers
            falsifying the mean square error and thus the log likelihood, the Hessian and the error bars.

        Returns
        -------
        1D-array
            array of size (n,) with the loss function values
        """    
        z = z0/threshold # we know the obj func is between 0 and 100 
        if loss=='linear':
            c = z
        elif loss=='soft_l1':
            c = 2 * ((1 + z)**0.5 - 1)
        elif loss=='huber':
            c =  z if z <= 1 else 2*z**0.5 - 1
        elif loss=='cauchy':
            c = np.log(1 + z)
        elif loss=='arctan':
            c = np.arctan(z)
        else:
            raise ValueError('Loss function not recognized (choose from linear, soft_l1, huber, cauchy, arctan)')
        return threshold*c # make sure the cost is between 0 and 100 too

    def invert_lossfunc(self,z0,loss,threshold=1000):
        """Invert the loss function to get the mean squared error values back

        Parameters
        ----------
        z0 : 1D-array
            data array of size (n,) with the loss function values
        loss : str
            type of loss function to use, can be ['linear','soft_l1','huber','cauchy','arctan']
        threshold : int, optional
            critical value above which loss sets in, by default 1000. 
            Wisely select so that the loss function affects only outliers.
            If threshold is set too low, then c<z0 even fon non-outliers
            falsifying the mean square error and thus the log likelihood, the Hessian and the error bars.

        Returns
        -------
        1D-array
            array of size (n,) with the mean squared error values
        """    
        z = z0/threshold
        if loss=='linear':
            c = z*threshold
        elif loss=='soft_l1':
            c = ((z/2+1)**2 - 1)*threshold
        elif loss=='huber':
            c = z*threshold if z <= 1 else ((z+1)**2/4)*threshold
        elif loss=='cauchy':
            c = (np.exp(z)-1)*threshold
        elif loss=='arctan':
            c = np.tan(z)*threshold
        else:
            raise ValueError('Loss function not recognized (choose from linear, soft_l1, huber, cauchy, arctan)')
        return c
    
    def format_func(self,value, tick_number):
        """Format function for the x and y axis ticks  
        to be passed to axo[ii,jj].xaxis.set_major_formatter(plt.FuncFormatter(format_func))  
        to get the logarithmic ticks looking good on the plot  

        Parameters
        ----------
        value : float
            value to convert
        tick_number : int
            tick position

        Returns
        -------
        str
            string representation of the value in scientific notation
        """        
        return sci_notation(10**value, sig_fig=-1)


    ###############################################################################
    ################################# Curve fit ###################################
    ###############################################################################

    def obj_func_curvefit(self,X,*p,params,model):
        """Objective function as desired by scipy.curve_fit

        Parameters
        ----------
        X : ndarray
            X Data array of size(n,m): n=number of data points, m=number of dimensions
        *p : ndarray
            list of float; values of the fit parameters as supplied by the optimizer
        params : list
            list of Fitparam() objects
        model : callable
            Model function yf = f(X) to compare to y

        Returns
        -------
        1D-array
            array of size (n,) with the model values
        """ 

        yfit = []
        for t in self.targets:
            xdata = t['data']['X'] # get array of experimental X values
            ydata = t['data']['y'] # get vector of experimental y values


            self.params_w(p, params) # write variable parameters into params, respecting user settings
            y = list(t['model'](xdata,params))
            yfit = yfit + y

        yfit = np.array(yfit)
        #print('errsq', np.sum(((yfit-self.yfull)/self.weightfull))**2)
        return yfit

    def optimize_curvefit(self, kwargs = None):
        """Use curvefit to optimize a function y = f(x) where x can be multi-dimensional
        use this function if the loss function is deterministic
        do not use this function if loss function has uncertainty (e.g from numerical simulation)
        in this case, use optimize_sko

        Parameters
        ----------
        kwargs : dict, optional
            kwargs aruguments for curve_fit, see scipy.optimize.curve_fit documentation for more information, by default None\\
            If no kwargs is provided use:\\
            kwargs = {'ftol':1e-8, 'xtol':1e-6, 'gtol': 1e-8, 'diff_step':0.001,'loss':'linear','max_nfev':5000}

        Returns
        -------
        dict
            dictionary with the optimized parameters ('popt') and the corresponding covariance ('pcov') and standard deviation ('std') values
        """  
        xfull = []
        yfull = []
        weightfull = []
        

        for t in self.targets:
            xdata = t['data']['X'] # get array of experimental X values
            ydata = list(t['data']['y']) # get vector of experimental y values
            if type(t['weight']) == int: # if weight is a number
                weight = t['weight'] * np.ones(len(ydata)) # set weight to a vector of the same size as ydata
            else:
                weight = list(1/np.asarray(t['weight']))
            # check if weight is a number or a vector
            weight = np.asarray(weight)
            if len(xfull)==0:
                xfull = xdata
                yfull = ydata 
                weightfull = list(1/weight)
            else:
                xfull = np.vstack((xfull,xdata))
                yfull = yfull + ydata
                weight = list(1/weight)
                weightfull = weightfull + weight
                
            if kwargs == None:
                kwargs = {'ftol':1e-8, 'xtol':1e-6, 'gtol': 1e-8, 'diff_step':0.001,'loss':'linear','max_nfev':5000}


        p0,lb,ub = self.params_r(self.params) # read out Fitparams & respect settings    
        # using partial functions allows passing extra parameters
        pfunc = partial(self.obj_func_curvefit, params = self.params, model = t['model'])

        r = curve_fit(pfunc,xfull,yfull,p0=p0,sigma=weightfull,absolute_sigma=False,check_finite=True,bounds=(lb,ub),full_output=True,method=None,jac=None, **kwargs)
        popt = r[0]
        pcov = r[1]
        infodict = r[2]
        mesg = r[3]
        std = np.sqrt(np.diag(pcov))
        
        # make tuple with (std,std) for each parameter
        stdx = []
        idx = 0
        for par in self.params:
            if par.relRange != 0:
                if par.optim_type == 'log':
                    p = par.val
                    s = std[idx]
                    s1 = 10**(np.log10(p)+s) - p # upper error bars
                    s2 = -10**(np.log10(p)-s) + p # lower error bars 
                    stdx.append((s2,s1))
                else:
                    s = std[idx]
                    stdx.append((s,s))
                idx += 1

        self.params_w(popt,self.params,std=stdx) # write variable parameters into params, respecting user settings

        return {'popt':popt,'pcov':pcov,'std':std,'infodict':infodict}

    ###############################################################################
    ########################## scipy Optimizers ###################################
    ###############################################################################

    def obj_func_scipy(self, *p, params, targets,obj_type='MSE', loss='linear'):
        """Objective function directly returning the loss function
        for use with the Bayesian Optimizer
        
        Parameters
        ----------
        '*p' : 1D-sequence of floats
            the parameters as passed by the Bayesian Optimizer
        params : list
            list of Fitparam() objects
        model : callable
            Model function yf = f(X) to compare to y
        X : ndarray
            X Data array of size(n,m): n=number of data points, m=number of dimensions
        y : 1D-array
            data array of size (n,) to fit 
        obj_type : str, optional
            objective function type, can be ['MSE', 'RMSE', 'MSLE','nRMSE','MAE','larry','nRMSE_VLC'], by default 'MSE'  

                'MSE': mean squared error  
                'RMSE': root mean squared error  
                'MSLE': mean squared log error  
                'nRMSE': normalized root mean squared error (RMSE/(max(y,yf)-min(y,yf)))  
                'MAE': mean absolute error  
                'RAE': relative absolute error sum(abs(yf-y))/sum(mean(y)-y)  
                'larry': mean squared error legacy version  
                'nRMSE_VLC': normalized root mean squared error (RMSE/(max(y,yf)-min(y,yf))) for each experiment separately and then averaged over all experiments  
        loss : str, optional
            type of loss function to use, can be ['linear','soft_l1','huber','cauchy','arctan'], by default 'linear'
        

        Returns
        -------
        1D-array
            array of size (n,) with the loss function values
        """

        px = p[0] # scikit-optimize provide 2d array of X values 
        self.params_w(px, params) # write variable parameters into params, respecting user settings
        zs = [] # list of losses
        target_weights = []
        for num, t in enumerate(targets):
            X = t['data']['X']
            y = t['data']['y']
            if 'weight' in t.keys(): 
                weight = t['weight']
            else:
                weight = 1
            yf = t['model'](X,params)
            # yf = model(X,params)

            if 'loss' in t.keys(): # if loss is specified in targets, use it
                loss_dum = t['loss']
            else:
                loss_dum = loss
            
            if 'obj_type' in t.keys():
                obj_type_dum = t['obj_type']
            else:
                obj_type_dum = obj_type
            
            z = self.obj_func_metric(t,yf,obj_type=obj_type_dum)

            if 'target_weight' in t.keys() and (isinstance(t['target_weight'], float) or isinstance(t['target_weight'], int)): # if target_weight is specified in targets, use it
                #check if target_weight is a float
                # if isinstance(t['target_weight'], float) or isinstance(t['target_weight'], int):
                    target_weights.append(t['target_weight'])
            else:
                target_weights.append(1)
                warnings.warn('target_weight for target '+str(num)+' must be a float or int. Using default value of 1.')

            zs.append(self.lossfunc(z,loss_dum,threshold=1))

        # cost is the weigthed average of the losses
        cost = np.average(zs, weights=target_weights)    

        return cost

    def optimize_dual_annealing(self, obj_type='MSE',loss='linear', kwargs = None):
        """Use dual_annealing to optimize a function y = f(x) where x can be multi-dimensional.
        See scipy.optimize.dual_annealing documentation for more information.

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments for dual_annealing, see scipy.optimize.dual_annealing documentation for more information, by default None.
            If no kwargs is provided, use:
            kwargs = {'maxiter': 100, 'local_search_options': None, 'initial_temp': 5230.0, 'restart_temp_ratio': 2e-05, 'visit': 2.62, 'accept': -5.0, 'maxfun': 1000, 'no_local_search': False, 'x0': None, 'bounds': None, 'args': ()}

        Returns
        -------
        dict
            Dictionary with the optimized parameters ('x') and the corresponding function value ('fun').
        """
        

        if kwargs == None:
            kwargs = {'maxiter': 100, 'local_search_options': None, 'initial_temp': 5230.0, 'restart_temp_ratio': 2e-05, 'visit': 2.62, 'accept': -5.0, 'maxfun': 1000, 'no_local_search': False, 'args': ()}


        p0,lb,ub = self.params_r(self.params) # read out Fitparams & respect settings
        # using partial functions allows passing extra parameters
        pfunc = partial(self.obj_func_scipy, params = self.params,targets=self.targets,obj_type=obj_type,loss=loss)

        result = dual_annealing(pfunc,x0=p0, bounds=list(zip(lb, ub)), **kwargs)
        popt = result.x
        fun = result.fun

        self.params_w(popt, self.params)  # write variable parameters into params, respecting user settings

        return {'x': popt, 'fun': fun}

    def optimize_basin_hopping(self, obj_type='MSE',loss='linear', kwargs = None):
        """Use basin_hopping to optimize a function y = f(x) where x can be multi-dimensional.
        See scipy.optimize.basin_hopping documentation for more information.

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments for basin_hopping, see scipy.optimize.basin_hopping documentation for more information, by default None.
            If no kwargs is provided, use:
            kwargs = {'niter': 100, 'T': 1.0, 'stepsize': 0.5, 'interval': 50, 'minimizer_kwargs': None, 'take_step': None, 'accept_test': None, 'callback': None, 'niter_success': None, 'seed': None, 'disp': False, 'niter_success': 10}

        Returns
        -------
        dict
            Dictionary with the optimized parameters ('x') and the corresponding function value ('fun').
        """
        if kwargs == None:
            kwargs = {'niter': 100, 'T': 1.0, 'stepsize': 0.5, 'interval': 50, 'minimizer_kwargs': None, 'take_step': None, 'accept_test': None, 'callback': None, 'niter_success': None, 'seed': None, 'disp': False, 'niter_success': 10}

        p0,lb,ub = self.params_r(self.params)
        # using partial functions allows passing extra parameters
        pfunc = partial(self.obj_func_scipy, params = self.params,targets=self.targets,obj_type=obj_type,loss=loss)
        
        # if 'minimizer_kwargs' in kwargs.keys() or kwargs['minimizer_kwargs'] == None:
        #     minimizer_kwargs = {"method": "L-BFGS-B", "bounds": list(zip(lb, ub))}
        #     kwargs['minimizer_kwargs'] = minimizer_kwargs

        result = basinhopping(pfunc, x0=p0, minimizer_kwargs={"bounds": list(zip(lb, ub))}, **kwargs)
        popt = result.x
        fun = result.fun

        self.params_w(popt, self.params)  # write variable parameters into params, respecting user settings

        return {'x': popt, 'fun': fun}



    