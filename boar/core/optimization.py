######################################################################_torch
################ MultiObjectiveOptimizer class #######################
######################################################################
# Authors: 
# Larry Lueer (https://github.com/larryluer)
# Vincent M. Le Corre (https://github.com/VMLC-PV)
# (c) i-MEET, University of Erlangen-Nuremberg, 2021-2022-2023 

# Import libraries
import itertools,matplotlib,os,json,copy,warnings
import numpy as np
import pandas as pd
from time import time
from functools import partial
from matplotlib import pyplot as plt
from matplotlib import cm
from copy import deepcopy
from tqdm import tnrange, tqdm_notebook
from scipy.optimize import curve_fit
from scipy.special import logsumexp
from scipy.ndimage.filters import gaussian_filter
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_objective 
from skopt.utils import expected_minimum
from joblib import Parallel, delayed
from types import SimpleNamespace
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

# Import boar libraries
from boar.core.optimizer import BoarOptimizer # import the base class


class MultiObjectiveOptimizer(BoarOptimizer):
    # a class for multi-objective optimization
    def __init__(self,params = None, targets = None, warmstart = None, Path2OldXY = None, SaveOldXY2file = None, res_dir = 'temp', parallel = True, verbose = False) -> None:
        """Initialization routine

        Parameters
        ----------
        targets : list of dict, optional
            List of Dictionaries with the following keys:\\
                'model': a pointer to a function y = f(X) where X has m dimensions\\
                'data': dictionary with keys\\
                'X':ndarray with shape (n,m) where n is the number of evaluations for X \\
                'y':ndarray with shape (n,)\\
                'X_dimensions': list of string: the names of the dimensions in X\\
                'X_units': list of string: the units of the dimensions in X\\
                'y_dimension': string: the name of the dimension y\\
                'y_unit': string: the unit of the dimension y\\
                'params': list of Fitparam() objects, by default None\\
        warmstart : str, optional
                'None', 'collect', 'collect_init', 'collect_BO', 'recall' or 'collect&recall', by default None\\
                    if 'None' does not store results\\
                    if 'collect' stores all results\\
                    if 'collect_init' stores results from the initial sampling only. NOTE: In reality it collects the n_jobs*(n_initial_points + n_BO)/n_initial_points first point, so you may get some of the first few BO points depending on n_jobs\\
                    if collect_BO' stores results from the Bayesian Optimization only\\
                    if 'recall' recalls results\\
                    if 'collect&recall' stores and recalls results\\
        Path2OldXY : str, optional
            full path to the json file containing the self.old_xy results to be loaded, by default None
        SaveOldXY2file : str, optional
            full path to the json file where the new self.old_xy results will be saved if it is None then we do not save the data, by default None
        res_dir : str, optional
            path to the directory where the results will be saved, by default 'temp'
        parallel : bool, optional
            use parallelization, if False n_jobs can still be > 1 but the evaluation will be done sequentially, by default True
        verbose : bool, optional
            print some information, by default False

        """        
        if targets != None:
            self.targets = targets
        
        if params != None:
            self.params = params
        
        self.warmstart = warmstart # want to re-use expensive calculations from a previous run?
        if self.warmstart == None:
            self.warmstart = 'None'
        
        self.old_xy = {'ydyn':1,'x':[],'y':[]} # dict to hold the calculations from a previous run if warmstart==True
        self.Path2OldXY = Path2OldXY
        self.SaveOldXY2file = SaveOldXY2file # path to file where we want to save the old_xy data
        self.verbose = verbose
        self.res_dir = res_dir
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)
        self.cwd = os.getcwd()

        self.parallel = parallel # do we want to run in parallel?
    
    
        
    def LLH(self,X,beta_scaled,N,gpr,fscale): # now that beta is known, get negative log likelihood
        """Return the negative log likelihood -ln(p(t|w,beta)) where t are the measured target values,
        w is the set of model parameters, and beta is the target uncertainty

        Ref.: Christopher M. Bishop:Pattern Recognition and Machine Learning, Springer Information Science & statistics 2006
        Chapter 1.2.5 pg 29

        Parameters
        ----------
        X : ndarray
            X Data array of size(n,m): n=number of data points, m=number of dimensions
        beta_scaled : float
            1 / minimum of scaled surrogate function
            will be multiplied with fscale to give 1/ the unscaled minimum which is the MSE of the best fit
            If there are no systematic deviations, and if the noise in y is Gaussian, then this should yield
            the variance of the Gaussian distribution of target values
        N : integer
            number of data points
        gpr : scikit-optimize estimator
            a trained regressor which has a .predict() method
        fscale : float
            scaling factor to keep the surrogate function between 0 and 100 (yields best results in BO but
            here we need the unscaled surrogate function, that is, MSE)

        Returns
        -------
        float
            the negative log likelihood
        """

        Y = gpr.predict([X], return_std=False)
        Y = Y[0] / fscale # remove the scaling to yield the MSE
        beta = beta_scaled * fscale # remove the scaling from beta too
        SSE = Y*N # the sum of squared errors
        LLH = -beta/2 * SSE + N/2*np.log(beta) - N/2*np.log(2*np.pi) # Bishop eq. 1.62
        return -LLH 
        
    def LH(self,X,beta_scaled,N,gpr,fscale):
        """Compute the positive log likelihood from the negative log likelihood

        Parameters:
        X : ndarray
            X Data array of size(n,m): n=number of data points, m=number of dimensions
        beta_scaled : float
            1 / minimum of scaled surrogate function
            will be multiplied with fscale to give 1/ the unscaled minimum which is the MSE of the best fit
            If there are no systematic deviations, and if the noise in y is Gaussian, then this should yield
            the variance of the Gaussian distribution of target values
        N : integer
            number of data points
        gpr : scikit-optimize estimator
            a trained regressor which has a .predict() method
        fscale : float
            scaling factor to keep the surrogate function between 0 and 100 (yields best results in BO but
            here we need the unscaled surrogate function, that is, MSE)

        Returns
        -------
        float
            the positive log likelihood
        """
        if len(fscale)>1: # we have more than one target
            fscale = 1 # we don't scale the likelihood for multiple targets since there is no way to the the MSE nicely if the values for the different targets ys are not comparable
        else:
            fscale = fscale[0] # we have only one target so we can keep the scaling
        Y = gpr.predict(X, return_std=False)
        Y = Y / fscale # remove the scaling to yield the MSE
        beta = beta_scaled * fscale # remove the scaling from beta too
        SSE = Y*N # the sum of squared errors
        LLH = -beta/2 * SSE + N/2*np.log(beta) - N/2*np.log(2*np.pi) # Bishop eq. 1.62
        #return np.exp(LLH)
        return LLH #
    
    ###############################################################################
    ############################ Posterior utils ##################################
    ###############################################################################

    def do_grid_posterior(self,step,fig,axes,gs,lb,ub,pf,beta_scaled, N, gpr, fscale, Nres, logscale, vmin, min_prob=1e-2, clear_axis=False,True_values=None,points=None):
        """Obtain the posterior probability distribution p(w|y) by brute force gridding
        where w is the set of model parameters and y is the data
        For each fitparameter, return mean and standard deviation

        Parameters
        ----------
        step : int
            Number of zooming steps
        fig : matplotlib.figure.Figure
            Figure to plot on
        axes : list
            List of axes to plot on
        gs : matplotlib.gridspec.GridSpec
            Gridspec to plot on
        lb : list
            Lower bounds of the grid
        ub : list
            Upper bounds of the grid
        pf : list
            List of parameters
        N : integer
            number of datasets
        gpr : scikit-optimize estimator
            trained regressor
        fscale : float
            scaling factor 
        Nres : integer
            Sampling resolution. Number of data points per dimension.
        logscale : boolean
            display in log scale?
        vmin : float
            lower cutoff (in terms of exp(vmin) if logscale==True)
        zoom : int, optional
            number of time to zoom in, by default 1.
        min_prob : float, optional
            minimum probability to consider when zooming in we will zoom on the parameter space with a probability higher than min_prob, by default 1e-40.
        clear_axis : boolean, optional
            clear the axis before plotting the zoomed in data, by default False.
        True_values : dict, optional
            dictionary of true values of the parameters, by default None
        points : array, optional
            array of explored points in the parameter space during the optimization, by default None
        Returns
        -------
        _type_
            _description_
        """ 

        pnames = [pp.name for pp in pf if pp.relRange!=0]
        # pnames_display = [pp.display_name for pp in pf if pp.relRange!=0] # for plots axis labels
        pnames_main = [pp.name for pp in pf]
        pnames_display = [pp.full_name for pp in pf if pp.relRange!=0]

        axis_type = [pp.axis_type for pp in pf if pp.relRange!=0] # to get the type of the axis for the plots
        optim_type = [pp.optim_type for pp in pf if pp.relRange!=0] # to get the type of the axis for the plots
        pval = [pp.val for pp in pf if pp.relRange!=0]
        # get varied fitparams
        # pf = [pp for pp in self.targets[-1]['params'] if pp.relRange!=0]
        p0, lb_main, ub_main = self.params_r(pf)

        # make sure that all the pnames are in the true values
        plot_true_values = True
        if True_values is not None:
            for pp in pnames:
                if pp not in True_values.keys():
                    plot_true_values = False
                    print('WARNING: true values not provided for all parameters. True values will not be plotted')
                    break
        else:
            plot_true_values = False
            if self.verbose:
                print('WARNING: true values not provided. True values will not be plotted')
            # if verbose:
            #     print('WARNING: true values not provided. True values will not be plotted')

        plot_points = False
        if points is not None:
            plot_points = True
            # if points.shape[1] != len(pnames):
            #     plot_points = False
            #     print('WARNING: points provided do not have the same dimension as the parameters. Points will not be plotted')

        # create grid
        dims,dims_GP = [],[]
        Nreso = Nres 
        # create grid and reconvert the parameters to the original scale
        for ii,pff in enumerate(pf): 
            if pff.relRange != 0:
                parax = np.linspace(lb[ii],ub[ii],Nreso)
                # add the best fit value if provided
                if pf[ii].optim_type == 'log':
                    best_val = np.log10(pf[ii].val)
                else:
                    best_val = pf[ii].val/pf[ii].p0m
                
                # put best value in par_ax and sort
                parax = np.sort(np.append(parax,best_val))


                if pff.optim_type == 'linear':
                    dum_lb = lb[ii] * pff.p0m
                    dum_ub = ub[ii] * pff.p0m
                    # dims.append(np.linspace(dum_lb,dum_ub,Nreso))
                    dims.append(parax*pff.p0m)

                elif pff.optim_type == 'log':
                    dum_lb = 10**(lb[ii])
                    dum_ub = 10**(ub[ii])
                    # dims.append(np.geomspace(dum_lb,dum_ub,Nreso))
                    dims.append(10**(parax))
                else: 
                    raise ValueError('ERROR. ',pff.name,' optim_type needs to be ''linear'' or ''log'' not ',pff.optim_type,'.')

                # dims_GP.append(np.linspace(lb[ii],ub[ii],Nreso))   #need as an input for the GP
                dims_GP.append(parax)   #need as an input for the GP
            
 
        if logscale: # if we are in logscale, we need to transform the grid
            min_prob = np.log10(min_prob)
            vmin = np.log10(vmin)

        # get the main grid
        p0, lb_old, ub_old = self.params_r(pf) # get the main parameter grid bounds
        # initialize figure
        nb_params = len(pf)
        
        # build complete matrix
        tic = time()
        XC = np.array(list(itertools.product(*dims_GP)))

        ## Make sure that len(Xc) < 1e7 otherwise we risk to fill up the RAM
        if len(XC) > 1e6:
            Xcc = np.array_split(XC, int(len(XC)/1e6))
            for ii in range(int(len(XC)/1e6)):
                if ii == 0:
                    aa0prime = self.LH(Xcc[ii],beta_scaled=beta_scaled,N=N,gpr=gpr,fscale=fscale)
                else:
                    aa0prime = np.concatenate((aa0prime, self.LH(Xcc[ii],beta_scaled=beta_scaled,N=N,gpr=gpr,fscale=fscale)))
            
            aa1 = aa0prime.reshape(*[Nreso+1 for _ in range(len(pf))]) # bring it into array format
            # Note: the Nreso plus one comes from adding the best fit value
        else:
            aa0 = self.LH(XC,beta_scaled=beta_scaled,N=N,gpr=gpr,fscale=fscale)# predict them all at once
            aa1 = aa0.reshape(*[Nreso+1 for _ in range(len(pf))]) # bring it into array format
            # Note: the Nreso plus one comes from adding the best fit value
        # build the single 2D panels
        
        for comb in list(itertools.combinations(range(nb_params), 2)):
            d1, d2 = comb[0], comb[1] 

            margax = [dd for dd in range(nb_params) if dd not in [d1,d2]] # the axes over which to marginalize
            ''' old way
            margax = [dd for dd in range(nb_params) if dd not in [d1,d2]] # the axes over which to marginalize
            LHS = logsumexp(aa1,axis = tuple(margax))
            # LHS-=np.max(LHS) # avoid underflow by setting the max to zero
            
            # if logscale:
            #     LHS[LHS<vmin]=vmin
            #     vmin = vmin
            #     vmax = 0
            # else:
            #     LHS = np.exp(LHS) # do the exponential => now it's prop. to probability
            #     # LHS/=np.sum(LHS)
            #     vmin = 0
            #     # vmax = 1
            #     vmax = np.max(LHS)

            # test 
            LHS = np.exp(LHS) # do the exponential => now it's prop. to probability
            total = np.sum(LHS)
            LHS/=total

            ######################
            # LHS = np.exp(aa1) # do the exponential => now it's prop. to probability
            # prob = np.sum(LHS,axis=tuple(margax))
            # prob = prob/np.sum(prob) # normalize to 1
            # LHS = prob
            ######################
            '''

            # Calculate the marginal posterior probability distribution p(w|y) for parameter ii by integrating over the other parameters
            lhlog = logsumexp(aa1,axis=tuple(margax)) # logsumexp is more accurate than np.log(np.sum(np.exp(lh),axis=1))
            p = np.exp(lhlog-logsumexp(lhlog)) #normalize the likelihood
            LHS = p
            
            if logscale:
                LHS = np.log10(LHS)
                LHS[LHS<vmin]=vmin
                vmin = vmin
                vmax = 0
            else:
                vmin = 0
                vmax = 1


            if clear_axis: # clear the axes for the zoomed in version
                axes[d2][d1].clear() # clear the previous plot
            
            
            axes[d2][d1].contourf(dims[d1],dims[d2],LHS.T,levels=50,vmin=vmin, vmax=vmax)
            axes[d1][d2].axis('off') # remove the upper triangle
            
            
            # plot the points
            if plot_points:
                pred1_plot = [x[d1] for x in points]
                pred2_plot = [x[d2] for x in points]
                # convert back to the right values  
                if optim_type[d1] == 'log':
                    pred1_plot = 10**(np.asarray(pred1_plot))
                elif optim_type[d1]== 'linear':
                    pred1_plot = np.asarray(pred1_plot) * pf[d1].p0m
                else:
                    raise ValueError('ERROR. ',pnames[d1],' optim_type needs to be ''linear'' or ''log'' not ',pf[d1].optim_type,'.')

                if optim_type[d2] == 'log':
                    pred2_plot = 10**(np.asarray(pred2_plot))

                elif optim_type[d2]== 'linear':
                    pred2_plot = np.asarray(pred2_plot) * pf[d2].p0m
                else:
                    raise ValueError('ERROR. ',pnames[d2],' optim_type needs to be ''linear'' or ''log'' not ',pf[d2].optim_type,'.')

                axes[d2][d1].plot(pred1_plot,pred2_plot,'o',color='k',markersize=3)
            
            # plot the true values
            try:
                if plot_true_values: 
                    axes[d2][d1].plot(True_values[pnames[d1]],True_values[pnames[d2]],'*',color='C3',markersize=10)
            except Exception as e:
                warnings.warn('WARNING. Could not plot the true values either '+pnames[d1]+' or '+pnames[d2]+' are not in the True_values dictionary.')
            # plot the best fit values
            axes[d2][d1].plot(pf[d1].val,pf[d2].val,'C2X',markersize=10)

            # Prepare the axis labels and ticks
            if step == 0 or clear_axis:
                if d1==0:
                    axes[d2][d1].set_ylabel(pnames_display[d2])
                    if axis_type[d2] == 'log': # set the ticks to be in log scale
                        axes[d2][d1].set_yscale('log')
                else:
                    if axis_type[d2] == 'log': # set the ticks to be in log scale
                        axes[d2][d1].set_yscale('log')
                    axes[d2][d1].tick_params(axis='y',labelleft=False,which='both') # remove the ticks


                if d2==nb_params-1:
                    axes[d2][d1].set_xlabel(pnames_display[d1])
                    axes[d2][d1].tick_params(axis='x', labelrotation = 45, which='both')
                    if axis_type[d1] == 'log': # set the ticks to be in log scale
                        axes[d2][d1].set_xscale('log')
                else:
                    if axis_type[d1] == 'log': # set the ticks to be in log scale
                        axes[d2][d1].set_xscale('log')
                    axes[d2][d1].tick_params(axis='x',labelbottom=False,which='both') # remove the ticks
                
                # set the limits
                if optim_type[d1] == 'linear':
                    axes[d2][d1].set_xlim((lb[d1]*pf[d1].p0m,ub[d1]*pf[d1].p0m))
                elif optim_type[d1] == 'log':
                    axes[d2][d1].set_xlim((10**(lb[d1]),10**(ub[d1])))
                else:
                    raise ValueError('ERROR. ',pnames[d1],' optim_type needs to be ''linear'' or ''log'' not ',optim_type[d1],'.')
                
                if optim_type[d2] == 'linear':
                    axes[d2][d1].set_ylim((lb[d2]*pf[d2].p0m,ub[d2]*pf[d2].p0m))
                elif optim_type[d2] == 'log':
                    axes[d2][d1].set_ylim((10**(lb[d2]),10**(ub[d2])))
                else:
                    raise ValueError('ERROR. ',pnames[d2],' optim_type needs to be ''linear'' or ''log'' not ',optim_type[d2],'.')
            else: # keep main axis limits
                # set the limits
                if optim_type[d1] == 'linear':
                    axes[d2][d1].set_xlim((lb_main[d1]*pf[d1].p0m,ub_main[d1]*pf[d1].p0m))
                elif optim_type[d1] == 'log':
                    axes[d2][d1].set_xlim((10**(lb_main[d1]),10**(ub_main[d1])))
                else:
                    raise ValueError('ERROR. ',pnames[d1],' optim_type needs to be ''linear'' or ''log'' not ',optim_type[d1],'.')

                if optim_type[d2] == 'linear':
                    axes[d2][d1].set_ylim((lb_main[d2]*pf[d2].p0m,ub_main[d2]*pf[d2].p0m))
                elif optim_type[d2] == 'log':
                    axes[d2][d1].set_ylim((10**(lb_main[d2]),10**(ub_main[d2])))
                else:
                    raise ValueError('ERROR. ',pnames[d2],' optim_type needs to be ''linear'' or ''log'' not ',optim_type[d2],'.')

                 
        # finally, build the 1D panels
        # in doing so, collect also the means and stds to write back to params
        xx,stdx = [],[]
        # initialize the new bounds
        
        lb_new = copy.deepcopy(lb)
        ub_new = copy.deepcopy(ub)

        # if step == 0:
        idx = 0
        for u,l in zip(ub,lb): # Make sure that bounds are put at the right order of magnitude 
            if pf[idx].optim_type == 'linear':
                ub_new[idx] *= pf[idx].p0m
                lb_new[idx] *= pf[idx].p0m
            elif pf[idx].optim_type == 'log':
                ub_new[idx] = 10**(ub_new[idx]) 
                lb_new[idx] = 10**(lb_new[idx])
            else:
                raise ValueError('ERROR. ',pf[idx].name,' optim_type needs to be ''linear'' or ''log'' not ',pf[idx].optim_type,'.')
            idx+=1
        std95 = []
        for comb in range(nb_params):
            margax = [dd for dd in range(nb_params) if dd !=comb] # the axes over which to marginalize
            ''' old way
            # LHS = logsumexp(aa1,axis = tuple(margax))
            # LHS-=np.max(LHS) # avoid underflow by setting the max to zero
            # LHS = np.exp(LHS) # do the exponential => now it's prop. to probability
            '''

            lhlog = logsumexp(aa1,axis=tuple(margax)) # logsumexp is more accurate than np.log(np.sum(np.exp(lh),axis=1))
            p = np.exp(lhlog-logsumexp(lhlog)) #normalize the likelihood
            LHS = p


            if logscale:
                LHS[LHS<10**vmin]=10**vmin
                vmin = vmin
                vmax = 1
            else:
                vmin = 0
                vmax = 1
            
            axes[comb][comb].clear() # clear the previous plot
            axes[comb][comb].plot(dims[comb],LHS)

            # plot true values as a line
            if plot_true_values:
                try:
                    axes[comb][comb].axvline(True_values[pnames[comb]],color='C3',linestyle='--')
                    axes[comb][comb].plot(True_values[pnames[comb]],1,'C3*',markersize=10)
                
                except Exception as e:
                    warnings.warn('WARNING. Could not plot true value for '+pnames[comb])


            if logscale:
                cut_prob = 10**min_prob
            else:
                cut_prob = min_prob
                
            idx = 0
            while idx < len(dims[comb])-1:
                if LHS[idx+1] > cut_prob:
                    lb_new[comb] = dims[comb][idx]
                    break
                else:
                    idx += 1


            idx = 0
            while idx < len(dims[comb])-1:
                if LHS[len(dims[comb])-idx-2] > cut_prob:
                    ub_new[comb] = dims[comb][len(dims[comb])-1-idx]
                    break
                else:
                    idx += 1

            # prepare the axis labels and ticks       
            # if step == 0 or clear_axis:
            if comb==0:
                if logscale:
                    axes[comb][comb].set_yscale('log')
                    axes[comb][comb].set_ylabel('P(w|y)') 
                    # axes[comb][comb].set_ylim((10**vmin,2)) 
                    axes[comb][comb].set_yticks([10**vmin,10**(int(vmin/2)),1], minor=False)
                    
                else:
                    axes[comb][comb].set_ylabel('P(w|y)')
                    axes[comb][comb].set_yticks([0,0.5,1], minor=False) 
                    # axes[comb][comb].set_ylim((0,2))
                    # axes[comb][comb].set_ylim((0,2))    
            else:
                axes[comb][comb].tick_params(axis='y',labelleft=False,which='both')
                axes[comb][comb].tick_params(axis='y',labelright=True,which='major')
                if logscale:
                    axes[comb][comb].set_yscale('log')
                    # axes[comb][comb].set_ylim((10**vmin,2))
                    axes[comb][comb].set_yticks([10**vmin,10**(int(vmin/2)),1], minor=False)
                    
                else:
                    axes[comb][comb].set_yticks([0,0.5,1], minor=False)
                    # axes[comb][comb].set_ylim((0,2))

            if axis_type[comb] == 'log': # set the ticks to be in log scale
                    axes[comb][comb].set_xscale('log')

            if comb==nb_params-1:
                axes[comb][comb].set_xlabel(pnames_display[comb])
                axes[comb][comb].tick_params(axis='x', labelrotation = 45, which='both')
                
                
            else:
                axes[comb][comb].tick_params(axis='x',labelbottom=False,which='both') # remove the ticks

            # set same limits as in the 2D plots
            if clear_axis or step == 0:
                if optim_type[comb] == 'linear':
                    axes[comb][comb].set_xlim((lb[comb]*pf[comb].p0m,ub[comb]*pf[comb].p0m))
                elif optim_type[comb] == 'log':
                    axes[comb][comb].set_xlim((10**(lb[comb]),10**(ub[comb])))
                else:
                    raise ValueError('ERROR. ',pnames[comb],' optim_type needs to be ''linear'' or ''log'' not ',optim_type[comb],'.')
                    
            else: # keep main axis limits
                if optim_type[comb] == 'linear':
                    axes[comb][comb].set_xlim((lb_main[comb]*pf[comb].p0m,ub_main[comb]*pf[comb].p0m))
                elif optim_type[comb] == 'log':
                    axes[comb][comb].set_xlim((10**(lb_main[comb]),10**(ub_main[comb])))
                else:
                    raise ValueError('ERROR. ',pnames[comb],' optim_type needs to be ''linear'' or ''log'' not ',optim_type[comb],'.')



            # get moments of cumulative probability density
            # pd = LHS
            # # xpd = dims[comb] # do not delogarithmize as this would distort the pdf!
            # # print(dims[comb])
            # xpd = dims_GP[comb] # do not delogarithmize as this would distort the pdf!
            # # print(xpd)
            # if optim_type[comb] == 'linear':
            #     xpd/=pf[comb].p0m 
            
            # m1 = np.sum(pd*(xpd)**1) / np.sum(pd) # the mean
            # m2 = np.sum(pd*(xpd-m1)**2) / np.sum(pd) # the variance
            # xx.append(m1)
            # stdx.append(np.sqrt(m2))

            # get 95 % confidence interval
            # find closest value to best fit value
            #
            x_name = pnames[comb]
            idx_X_main = pnames_main.index(x_name)
            if pf[idx_X_main].optim_type == 'log':
                best_val = pf[idx_X_main].val
            else:
                best_val = pf[idx_X_main].val
            idx_best = np.argmin(np.abs(dims[comb]-best_val))

            par_ax = dims[comb]
            p = LHS
            ilow = 0
            ihigh = len(par_ax)-1
            sum_prob = 0
            for i in range(len(par_ax)-1):
                sum_prob += p[i]
                if sum_prob >= 0.025:
                    ilow = i
                    break
            sum_prob = 0
            for i in range(len(par_ax)-1):
                sum_prob += p[len(par_ax)-1-i]
                if sum_prob >= 0.025:
                    ihigh = len(par_ax)-1-i
                    break
            if ilow > idx_best:
                ilow = idx_best
            if ihigh < idx_best:
                ihigh = idx_best
            
            # 95% confidence interval centered on the best fit value
            ihigh = min(ihigh,len(par_ax)-1)
            ilow = max(ilow,0)

            limlow = min(par_ax[ilow],best_val)
            limhigh = max(par_ax[ihigh],best_val)

            # add the 95% confidence interval to the plot
            axes[comb][comb].axvline(limlow,color='C0',linestyle='-.')
            axes[comb][comb].axvline(limhigh,color='C0',linestyle='-.')
            std95.append((limlow,limhigh))
            # Note: this is the 95% confidence interval unless the ground truth val is too far from the minimum of the surrogate model
            # in that case one of the limits of the confidence interval is the ground truth value

            # color the 95% confidence interval
            mask = (par_ax >= limlow) & (par_ax <= limhigh) # Create mask for the shaded area
            axes[comb][comb].fill_between(par_ax, p, where=mask, color='C0', alpha=0.5) # Fill the area below the curve between the vertical lines

            axes[comb][comb].axvline(pf[idx_X_main].val,color='C0',linestyle='-') # add the best fit value to the plot

            # Set the axis limits and labels
            if pf[idx_X_main].optim_type == 'log':
                axes[comb][comb].set_xlim([10**(lb[comb]),10**(ub[comb])])
            else:
                axes[comb][comb].set_xlim([lb[comb]*pf[idx_X_main].p0m,ub[comb]*pf[idx_X_main].p0m])

        print('Sampling for posterior distribution done in ', time()-tic,'s')

        ## Make colorbar
        # Define the logarithmic space for the colorbar
        cmap = plt.get_cmap('viridis')
        if logscale:
            norm = matplotlib.colors.LogNorm(vmin=10**vmin, vmax=10**vmax)
            ticks = [10**vmin,10**(int(vmin/2)),1]
        else:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
            ticks = [0,0.5,1]

        # Create a scalar mappable to map the values to colors
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Create a colorbar
        if logscale:
            fmt = matplotlib.ticker.LogFormatterMathtext()
        else:
            fmt = None

        cax = fig.add_axes([0.75, 0.6, 0.05, 0.2])  # left, bottom, width, height
        cbar = plt.colorbar(sm, cax=cax, ticks=ticks, format=fmt)


        # Set the colorbar label on the left side
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.set_ylabel('P(w|y)', rotation=90, va='bottom')


        # rearrange the lb_new and ub_new for the next iteration

        for comb in range(nb_params):
            if pf[comb].optim_type == 'linear':
                lb_new[comb] = lb_new[comb]/pf[comb].p0m
                ub_new[comb] = ub_new[comb]/pf[comb].p0m

            elif pf[comb].optim_type == 'log':
                lb_new[comb] = np.log10(lb_new[comb])
                ub_new[comb] = np.log10(ub_new[comb])

            else:
                raise ValueError('ERROR. ',pnames[comb],' optim_type needs to be ''linear'' or ''log'' not ',optim_type[comb],'.')


        # return xx,stdx, lb_new, ub_new
        return xx,std95, lb_new, ub_new

    def posterior(self, params, lb_main, ub_main, beta_scaled, N, gpr, fscale,kwargs_posterior, points = None):#, Nres, logscale, vmin, zoom=1, min_prob=1e-40, clear_axis=False):
        """Obtain the posterior probability distribution p(w|y) by brute force gridding
        where w is the set of model parameters and y is the data
        For each fitparameter, return mean and standard deviation


        Parameters
        ----------
        params : list of fitparameter objects
            list of fitparameters
        lb_main : list of floats
            lower bound for the fitparameters
        ub_main : list of floats
            upper bound for the fitparameters
        beta_scaled : float
            1/minimum of the scaled surrogate function
        N : integer
            number of datasets
        gpr : scikit-optimize estimator
            trained regressor
        fscale : float
            scaling factor
        kwargs_posterior : dict
            dictionary of keyword arguments for posterior function
            including: 
                Nres : integer, optional
                    Sampling resolution. Number of data points per dimension, by default 30
                Ninteg : integer, optional
                    Number of points for the marginalization over the other parameters when full_grid = False, by default 100
                full_grid : boolean, optional
                    If True, use a full grid for the posterior, by default False
                randomize : boolean, optional
                    If True, calculate the posterior for all the dimension but draw the marginalization points randomly and do not use a corse grid, by default False
                logscale : boolean, optional
                    display in log scale?, by default True
                vmin : float, optional
                    lower cutoff (in terms of exp(vmin) if logscale==True), by default 1e-100
                zoom : int, optional
                    number of time to zoom in, only used if full_grid = True, by default 0
                min_prob : float, optional
                    minimum probability to consider when zooming in we will zoom on the parameter space with a probability higher than min_prob, by default 1e-40.
                clear_axis : boolean, optional
                    clear the axis before plotting the zoomed in data, by default False.
                True_values : dict, optional
                    dictionary of true values of the parameters, by default None
                show_points : boolean, optional
                    show the explored points in the parameter space during the optimization, by default False
                savefig : boolean, optional
                    save the figure, by default False
                savefig_name : str, optional
                    name of the file to save the figure, by default 'posterior.png'
                savefig_dir : str, optional
                    directory to save the figure, by default self.res_dir
                figext : str, optional
                    extension of the figure, by default '.png'
                figsize : tuple, optional
                    size of the figure, by default (5*nb_params,5*nb_params)
                figdpi : int, optional
                    dpi of the figure, by default 300
        points : array, optional
            array of explored points in the parameter space during the optimization, by default None
        

        Returns
        -------
        Contour plots for each pair of fit parameters
        list of float, list of float
           the mean and the square root of the second central moment 
           (generalized standard deviation for arbitrary probability distribution)
        """

        # get varied fitparams
        # pf = [pp for pp in self.targets[-1]['params'] if pp.relRange!=0]
        # p0, lb_main, ub_main = self.params_r(self.targets[-1]['params'])

        # initialize figure
        nb_params = len(params)

        # get kwargs
        Nres = kwargs_posterior.get('Nres',10)
        Ninteg = kwargs_posterior.get('Ninteg',1e5)
        full_grid = kwargs_posterior.get('full_grid',False)
        randomize = kwargs_posterior.get('randomize',False)
        logscale = kwargs_posterior.get('logscale',True)
        vmin = kwargs_posterior.get('vmin',1e-100)
        zoom = kwargs_posterior.get('zoom',0)
        min_prob = kwargs_posterior.get('min_prob',1e-40)
        clear_axis = kwargs_posterior.get('clear_axis',False)
        True_values = kwargs_posterior.get('True_values',None)
        show_points = kwargs_posterior.get('show_points',False)
        savefig = kwargs_posterior.get('savefig',False)
        figname = kwargs_posterior.get('figname','posterior')
        figdir = kwargs_posterior.get('figdir',self.res_dir)
        figext = kwargs_posterior.get('figext','.png')
        figsize = kwargs_posterior.get('figsize',(5*nb_params,5*nb_params))
        figdpi = kwargs_posterior.get('figdpi',300)
        show_fig = kwargs_posterior.get('show_fig',True)

        # save parameters to self for later use
        self.Nres = Nres
        self.Ninteg = Ninteg
        self.logscale = logscale
        self.vmin = vmin
        self.min_prob = min_prob
       


        if show_points == False:
            points = None
        

        if full_grid == True and randomize == False:

            fig, axes = plt.subplots(nrows=nb_params, ncols=nb_params, figsize=figsize)
            gs = GridSpec(nb_params, nb_params, figure=fig)
            if zoom > 0:
                for i in range(zoom):
                    if i == 0:
                        old_lb = copy.deepcopy(lb_main)
                        old_ub = copy.deepcopy(ub_main)
                    else:
                        old_lb = copy.deepcopy(new_lb)
                        old_ub = copy.deepcopy(new_ub)
                    
                    xx, stdx, new_lb, new_ub = self.do_grid_posterior(i,fig,axes,gs,old_lb,old_ub,params,beta_scaled, N, gpr, fscale, Nres, logscale, vmin,min_prob=min_prob, clear_axis=clear_axis, True_values=True_values, points=points)

                    if new_lb == old_lb and new_ub == old_ub:
                        print('Only {} zooms done'.format(i))
                        print('No more zooming in, to zoom in further, change the min_prob')
                        break
                    
                    if savefig:
                        fig.savefig(os.path.join(figdir,figname+'_zoom_'+str(i)+figext), dpi=figdpi)
                print('{} zooms done'.format(zoom))
                    
            else:
                xx, stdx, lb, ub = self.do_grid_posterior(0,fig,axes,gs,lb_main,ub_main,params,beta_scaled, N, gpr, fscale, Nres, logscale, vmin,min_prob=min_prob, clear_axis=clear_axis, True_values=True_values, points=points)

            if show_fig:
                plt.show()

            if savefig:
                plt.tight_layout()
                fig.savefig(os.path.join(figdir,figname+figext), dpi=300)

        elif full_grid == True and randomize == True:
            
            xx, stdx, lb, ub = self.randomize_grid_posterior(params, lb_main, ub_main, beta_scaled, N, gpr, fscale,kwargs_posterior, points = points,True_values=True_values)
        else:
            # make len(self.params)/2 x len(self.params)/2 grid of subplots
            free_params = [p for p in params if p.relRange != 0]
            num_plots = len(free_params)
            num_rows = int(num_plots ** 0.5)  # Calculate number of columns
            num_cols = int((num_plots + num_rows - 1) // num_rows) # Calculate number of rows
            if num_rows == 0:
                num_rows = 1
            if num_cols == 0:
                num_cols = 1
            f, axes = plt.subplots(num_rows, num_cols,figsize=figsize)
            idx = 0
            xx,stdx = [],[]
            for param in params:
                if param.relRange != 0:
                    row = int(idx // num_cols)
                    col = int(idx % num_cols)
                    if num_rows == 1:
                        ax = axes[col]
                    else:
                        ax = axes[row, col]
                    x_, std_, lb, ub = self.marginal_posterior_1D(param.name, pf=params, fig=f, ax=ax,True_values=True_values,points=points,logscale=logscale,lb = lb_main, ub = ub_main,beta_scaled = beta_scaled, N = N,gpr=gpr, fscale=fscale, Nres=Nres, Ninteg=Ninteg, vmin=vmin, min_prob=min_prob, clear_axis=clear_axis)
                    
                    idx += 1
                    xx.append(x_)
                    stdx.append(std_)
            f.tight_layout()

            if show_fig:
                plt.show()
            if savefig:
                f.savefig(os.path.join(figdir,figname+figext), dpi=300)

        return xx,stdx


    def marginal_posterior_1D(self, x_name, pf = None, lb = None, ub = None, fig = None, ax = None, True_values = None, gpr = None, N = None, beta_scaled =None, fscale = None, Nres = None, Ninteg = 1e5, vmin = None, min_prob=None, points = None, logscale = False, show_plot = True, clear_axis = False, xlabel_pos = 'bottom', ylabel_pos = 'left', **kwargs):

        """ calculate and plot the marginal posterior probability distribution p(w|y) for parameter x_name by integrating over the other parameters
        
        Parameters
        ----------
        x_name : str
            name of the parameter for which the marginal posterior probability distribution is calculated
        lb : float, optional
            lower bound of the parameter x_name, if None we use the main boundaries, by default None
        ub : float, optional
            upper bound of the parameter x_name, if None we use the main boundaries, by default None
        fig : matplotlib figure, optional
            figure to plot the marginal posterior probability distribution, if None we create a new figure, by default None
        ax : matplotlib axis, optional
            axis to plot the marginal posterior probability distribution, if None we create a new axis, by default None
        True_values : dict, optional
            dictionary with the true values of the parameters, by default None
        gpr : sklearn regressor, optional
            regressor to calculate the likelihood, if None we use the self.gpr, by default None
        N : int, optional
            number of samples to calculate the likelihood, if None we use the self.N, by default None
        beta_scaled : float, optional
            scaling factor for the likelihood, if None we use the self.beta_scaled, by default None
        fscale : float, optional
            scaling factor for the likelihood, if None we use the self.fscale, by default None
        Nres : int, optional
            number of points to calculate the marginal posterior probability distribution, by default None
        Ninteg : int, optional
            number of points to marginalize the prob, by default 1e5
        vmin : float, optional
            minimum value of the marginal posterior probability distribution, only used if logscale = True as for linscale the min probability is 0, by default None
        min_prob : float, optional
            value used for the cut off probability when zooming in, note that for now this is not in used, by default None
        points : array, optional
            array with the points to plot the marginal posterior probability distribution, by default None
        logscale : bool, optional
            if True we plot the marginal posterior probability distribution in log scale, by default False
        show_plot : bool, optional
            if True we show the plot, by default True
        clear_axis : bool, optional
            if True we clear the axis, by default False
        xlabel_pos : str, optional
            position of the xlabel, by default 'bottom'
        ylabel_pos : str, optional
            position of the ylabel, by default 'left'
        **kwargs : dict, optional
            additional arguments to pass to the plot function, by default None

        Returns
        -------

        """ 
        # Make sure we have all the parameters we need otherwise use the values in self    
        if pf is None:
            # check is self.pf is intialized
            if hasattr(self,'params') is True:
                pf = self.params
            else:
                raise ValueError("self.pf is not initialized and no pf is provided.")
        if gpr is None:
            # check is self.gpr is intialized
            if hasattr(self,'gpr') is True: 
                gpr = self.gpr
            else:
                raise ValueError("self.gpr is not initialized and no gpr is provided.")
        if N is None:
            # check is self.N is intialized
            if hasattr(self,'N') is True: 
                N = self.N
            else:
                raise ValueError("self.N is not initialized and no N is provided.")
        if beta_scaled is None:
            # check is self.beta_scaled is intialized
            if hasattr(self,'beta_scaled') is True:
                beta_scaled = self.beta_scaled
            else:
                raise ValueError("self.beta_scaled is not initialized and no beta_scaled is provided.")
        if fscale is None:
            # check is self.fscale is intialized
            if hasattr(self,'fscale') is False: 
                fscale = [1]
                warnings.warn("self.fscale is not initialized and no fscale is provided. Set to default value of 1.")
            else:
                fscale = self.fscale
        if Nres is None:
            # check is self.Nres is intialized
            if hasattr(self,'Ninteg') is False:
                Nres = int(10) 
                warnings.warn("self.Nres is not initialized and no Nres is provided. Set to default value of 10.")
            else:
                Nres = int(self.Nres) # make sure it is an integer
        else:
            Nres = int(Nres)
        if Ninteg is None:
            if hasattr(self,'Ninteg') is False: # number of samples to draw from the grid to calculate the likelihood
                Ninteg = int(1e5)
                warnings.warn("self.Ninteg is not initialized and no Ninteg is provided. Set to default value of 1e5.")
            else:
                Ninteg = int(self.Ninteg)
        else:
            Ninteg = int(Ninteg)
        if vmin is None:
            if logscale is True:
                if hasattr(self,'vmin') is False:
                    vmin = 1e-10
                    warnings.warn("self.vmin is not initialized and no vmin is provided. Set to default value of 1e-10.")
                else:
                    vmin = self.vmin

        pnames_main = [pp.name for pp in pf]
        pnames = [pp.name for pp in pf if pp.relRange!=0]
        pnames_display = [pp.display_name for pp in pf if pp.relRange!=0]
        pnames_full = [pp.full_name for pp in pf if pp.relRange!=0]

        # get the bounds of the parameters
        p0, lb_main, ub_main = self.params_r(pf) # get the main bounds of the parameters (i.e. with the zooming)
        if lb is None:
            lb = lb_main
        if ub is None:
            ub = ub_main

        
        # get the index of the parameter to plot in pnames_main
        idx_X_main = pnames_main.index(x_name)
        ii = pnames.index(x_name)

        # initialize the figure if not provided
        if (fig is None and ax is None) or (ax is None):
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111)
            not_init = True
        else:
            not_init = False

        # create a linspace for the parameter ii
        par_ax = np.linspace(lb[ii],ub[ii],Nres)
        # add the best fit value if provided
        if pf[idx_X_main].optim_type == 'log':
            best_val = np.log10(pf[idx_X_main].val)
        else:
            best_val = pf[idx_X_main].val/pf[idx_X_main].p0m

        # put best value in par_ax and sort
        par_ax = np.sort(np.append(par_ax,best_val))
        idx_best = np.where(par_ax==best_val)[0][0]#get best_val position

        # create an empty array to store the likelihood
        lh = np.zeros((len(par_ax),Ninteg))

        # make a 2D vector where the value of parameter ii is fixed to the values in par_ax and a Ninteg random samples are drawn randomly from the grid to set the values of the other parameters
        for i in range(len(par_ax)):
            X = np.zeros((Ninteg,len(lb)))
            X[:,ii] = par_ax[i]
            for jj in range(len(lb)):
                if jj!=ii:
                    X[:,jj] = np.random.uniform(lb[jj],ub[jj],Ninteg)
            lh[i,:] = self.LH(X,beta_scaled,N,gpr,fscale)

        # Calculate the marginal posterior probability distribution p(w|y) for parameter ii by integrating over the other parameters
        lhlog = logsumexp(lh,axis=1) # logsumexp is more accurate than np.log(np.sum(np.exp(lh),axis=1))
        p = np.exp(lhlog-logsumexp(lhlog)) #normalize the likelihood

       
        ''' old way (Not accurate so need logsumexp)
        lh[i,:] =  np.exp(self.LH(X,beta_scaled,N,gpr,fscale))
        calculate the marginal posterior probability distribution p(w|y) for parameter ii by integrating over the other parameters
        p = np.sum(lh,axis=1)
        p = p/np.sum(p)
        '''

        if np.sum(p) <= 1e-3: # this is in case all p are zeros 
            warnings.warn("The marginal posterior probability distribution is all zeros. We set it to a uniform distribution.")
            p = np.ones(len(p))

        # prepare the axis
        if pf[idx_X_main].optim_type == 'log':
            par_ax = 10**(par_ax)
        else:
            par_ax = par_ax * pf[idx_X_main].p0m

        # plot the marginal posterior probability distribution p(w|y) for parameter ii
        if logscale:
            p[p<vmin] = vmin
            p[np.isnan(p)] = vmin
        else:
            p[p<0] = 0
            # or or p is nan
            p[np.isnan(p)] = 0
        ax.plot(par_ax,p,'C0',linewidth=2)

        if pf[idx_X_main].axis_type == 'log':
            ax.set_xscale('log')
        
        ilow = 0
        ihigh = len(par_ax)-1
        sum_prob = 0
        for i in range(len(par_ax)-1):
            sum_prob += p[i]
            if sum_prob >= 0.025:
                ilow = i
                break
        sum_prob = 0
        for i in range(len(par_ax)-1):
            sum_prob += p[len(par_ax)-1-i]
            if sum_prob >= 0.025:
                ihigh = len(par_ax)-1-i
                break
        if ilow > idx_best:
            ilow = idx_best
        if ihigh < idx_best:
            ihigh = idx_best


        # 95% confidence interval centered on the best fit value
        ihigh = min(ihigh,len(par_ax)-1)
        ilow = max(ilow,0)
        # add the 95% confidence interval to the plot
        ax.axvline(par_ax[ilow],color='C0',linestyle='-.')
        ax.axvline(par_ax[ihigh],color='C0',linestyle='-.')

        # Note: this is the 95% confidence interval unless the ground truth val is too far from the minimum of the surrogate model
        # in that case one of the limits of the confidence interval is the ground truth value

        # color the 95% confidence interval
        mask = (par_ax >= par_ax[ilow]) & (par_ax <= par_ax[ihigh]) # Create mask for the shaded area
        ax.fill_between(par_ax, p, where=mask, color='C0', alpha=0.5) # Fill the area below the curve between the vertical lines

        ax.axvline(pf[idx_X_main].val,color='C0',linestyle='-') # add the best fit value to the plot

        try:
            if True_values is not None: # plot the true value of parameter ii if provided
            
                ax.axvline(True_values[pnames[ii]],color='C3',linestyle='--')
                ax.plot(True_values[pnames[ii]],1,'C3*',markersize=10)
        except Exception as e:
            warnings.warn('WARNING. Could not plot true value for '+pnames[ii])


        ''' Old way of calculating the mean and standard deviation of the marginal posterior probability distribution p(w|y) for parameter ii (not used anymore)
        calculate the mean and standard deviation of the marginal posterior probability distribution p(w|y) for parameter ii
        m1 = np.sum(p*par_ax) / np.sum(p) # the mean
        m2 = np.sum(p*(par_ax-m1)**2) / np.sum(p) # the variapar_axTnce
        std = np.sqrt(m2) # the standard deviation
        '''

        # Set the axis limits and labels
        if pf[idx_X_main].optim_type == 'log':
            ax.set_xlim([10**(lb[ii]),10**(ub[ii])])
        else:
            ax.set_xlim([lb[ii]*pf[idx_X_main].p0m,ub[ii]*pf[idx_X_main].p0m])

        ax.tick_params(axis='x', labelrotation = 45, which='both')

        if logscale:
            ax.set_yscale('log')
            ax.set_ylim([vmin,2])
        else:
            ax.set_ylim([0,1.1])

        if xlabel_pos is not None:
            ax.set_xlabel(pnames_full[ii],position = xlabel_pos)
            if xlabel_pos == 'top':
                ax.xaxis.set_label_position('top')
                ax.tick_params(axis='x',labelbottom=False,which='both')
                ax.tick_params(axis='x',labeltop=True,which='both')
            else:
                ax.xaxis.set_label_position('bottom')
                ax.tick_params(axis='x',labelbottom=True,which='both')
                ax.tick_params(axis='x',labeltop=False,which='both')

        else:
            ax.set_xticklabels([]) # remove the x label ticks  
            ax.set_xticks([]) # remove the x label ticks
            ax.set_xlabel('')
            ax.tick_params(axis='x',labelbottom=False,which='both')
            ax.tick_params(axis='x',labeltop=False,which='both')

        if ylabel_pos is not None:
            ax.set_ylabel('P('+pnames_display[ii]+'|y)',position = ylabel_pos)
            if ylabel_pos == 'right':
                ax.yaxis.set_label_position('right')
                ax.tick_params(axis='y',labelleft=False,which='both')
                ax.tick_params(axis='y',labelright=True,which='both')
            else:
                ax.yaxis.set_label_position('left')
                ax.tick_params(axis='y',labelleft=True,which='both')
                ax.tick_params(axis='y',labelright=False,which='both')
        else:
            ax.set_yticklabels([]) # remove the x label ticks  
            ax.set_yticks([]) # remove the x label ticks
            ax.set_ylabel('')
            ax.tick_params(axis='y',labelleft=False,which='both')
            ax.tick_params(axis='y',labelright=False,which='both')

        
        if not_init:
            fig.tight_layout()
            plt.show()

        # max probability
        idx_max = np.argmax(p)
        xmin = par_ax[idx_max]
        std = (par_ax[ilow],par_ax[ihigh]) # 95% confidence interval

        # Prepare new bounds for next zoom (not needed for now but maybe in the future)
        lb_new = copy.deepcopy(lb)
        ub_new = copy.deepcopy(ub)



        return xmin,std,lb_new,ub_new
    
    def marginal_posterior_2D(self, x_name, y_name, pf = None, lb = None, ub = None, fig = None, ax = None, True_values = None, gpr = None, N = None, beta_scaled =None, fscale = None, Nres = None, Ninteg = 1e5, vmin = None, min_prob=None, points = None, logscale = False, show_plot = True, clear_axis = False, xlabel_pos = 'bottom', ylabel_pos = 'left', **kwargs):

        """ calculate and plot the marginal posterior probability distribution p(w|y) for parameter x_name by integrating over the other parameters
        
        Parameters
        ----------
        x_name : str
            name of the parameter for which the marginal posterior probability distribution is calculated on the x-axis
        y_name : str
            name of the parameter for which the marginal posterior probability distribution is calculated on the y-axis
        lb : float, optional
            lower bound of the parameter x_name, if None we use the main boundaries, by default None
        ub : float, optional
            upper bound of the parameter x_name, if None we use the main boundaries, by default None
        fig : matplotlib figure, optional
            figure to plot the marginal posterior probability distribution, if None we create a new figure, by default None
        ax : matplotlib axis, optional
            axis to plot the marginal posterior probability distribution, if None we create a new axis, by default None
        True_values : dict, optional
            dictionary with the true values of the parameters, by default None
        gpr : sklearn regressor, optional
            regressor to calculate the likelihood, if None we use the self.gpr, by default None
        N : int, optional
            number of samples to calculate the likelihood, if None we use the self.N, by default None
        beta_scaled : float, optional
            scaling factor for the likelihood, if None we use the self.beta_scaled, by default None
        fscale : float, optional
            scaling factor for the likelihood, if None we use the self.fscale, by default None
        Nres : int, optional
            number of points to calculate the marginal posterior probability distribution, by default None
        Ninteg : int, optional
            number of points to marginalize the prob, by default 1e5
        vmin : float, optional
            minimum value of the marginal posterior probability distribution, only used if logscale = True as for linscale the min probability is 0, by default None
        min_prob : float, optional
            value used for the cut off probability when zooming in, note that for now this is not in used, by default None
        points : array, optional
            array with the points to plot the marginal posterior probability distribution, by default None
        logscale : bool, optional
            if True we plot the marginal posterior probability distribution in log scale, by default False
        show_plot : bool, optional
            if True we show the plot, by default True
        clear_axis : bool, optional
            if True we clear the axis, by default False
        xlabel_pos : str, optional
            position of the xlabel, by default 'bottom'
        ylabel_pos : str, optional
            position of the ylabel, by default 'left'
        **kwargs : dict, optional
            additional arguments to pass to the plot function, by default None
                show_points : bool, optional
                    if True we show the points, by default True

        Returns
        -------

        """ 

        show_points = kwargs.setdefault('show_points', True)

        # Make sure we have all the parameters we need otherwise use the values in self    
        if pf is None:
            # check is self.pf is intialized
            if hasattr(self,'params') is True:
                pf = self.params
            else:
                raise ValueError("self.pf is not initialized and no pf is provided.")
        if gpr is None:
            # check is self.gpr is intialized
            if hasattr(self,'gpr') is True: 
                gpr = self.gpr
            else:
                raise ValueError("self.gpr is not initialized and no gpr is provided.")
        if N is None:
            # check is self.N is intialized
            if hasattr(self,'N') is True: 
                N = self.N
            else:
                raise ValueError("self.N is not initialized and no N is provided.")
        if beta_scaled is None:
            # check is self.beta_scaled is intialized
            if hasattr(self,'beta_scaled') is True:
                beta_scaled = self.beta_scaled
            else:
                raise ValueError("self.beta_scaled is not initialized and no beta_scaled is provided.")
        if fscale is None:
            # check is self.fscale is intialized
            if hasattr(self,'fscale') is False: 
                fscale = [1]
                warnings.warn("self.fscale is not initialized and no fscale is provided. Set to default value of 1.")
            else:
                fscale = self.fscale
        if Nres is None:
            # check is self.Nres is intialized
            if hasattr(self,'Ninteg') is False:
                Nres = int(10) 
                warnings.warn("self.Nres is not initialized and no Nres is provided. Set to default value of 10.")
            else:
                Nres = int(self.Nres) # make sure it is an integer
        else:
            Nres = int(Nres)
        if Ninteg is None:
            if hasattr(self,'Ninteg') is False: # number of samples to draw from the grid to calculate the likelihood
                Ninteg = int(1e5)
                warnings.warn("self.Ninteg is not initialized and no Ninteg is provided. Set to default value of 1e5.")
            else:
                Ninteg = int(self.Ninteg)
        else:
            Ninteg = int(Ninteg)
        if vmin is None:
            if logscale is True:
                if hasattr(self,'vmin') is False:
                    vmin = 1e-10
                    warnings.warn("self.vmin is not initialized and no vmin is provided. Set to default value of 1e-10.")
                else:
                    vmin = self.vmin

        # if show_points is True and points is None:
        #     points = self.points

        
        
        pnames_main = [pp.name for pp in pf]
        pnames = [pp.name for pp in pf if pp.relRange!=0]
        pnames_display = [pp.display_name for pp in pf if pp.relRange!=0]
        pnames_full = [pp.full_name for pp in pf if pp.relRange!=0]

        # get the bounds of the parameters
        p0, lb_main, ub_main = self.params_r(pf) # get the main bounds of the parameters (i.e. with the zooming)
        if lb is None:
            lb = lb_main
        if ub is None:
            ub = ub_main

        
        # get the index of the parameter to plot in pnames_main
        idx_X_main = pnames_main.index(x_name)
        iiX = pnames.index(x_name)
        idx_Y_main = pnames_main.index(y_name)
        iiY = pnames.index(y_name)

        # initialize the figure if not provided
        if (fig is None and ax is None) or (ax is None):
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111)
            not_init = True
        elif fig is not None and ax is None:
            # get figure number
            fig_num = fig.number
            #activate figure
            plt.figure(fig_num)
            ax = fig.add_subplot(111)
            not_init = True
        else:
            not_init = False

        # create a linspace for the parameter ii
        par_ax = np.linspace(lb[iiX],ub[iiX],Nres)
        par_ay = np.linspace(lb[iiY],ub[iiY],Nres)

        # add the best fit value if provided
        if pf[idx_X_main].optim_type == 'log':
            best_val_x = np.log10(pf[idx_X_main].val)
        else:
            best_val_x = pf[idx_X_main].val/pf[idx_X_main].p0m
        
        if pf[idx_Y_main].optim_type == 'log':
            best_val_y = np.log10(pf[idx_Y_main].val)
        else:
            best_val_y = pf[idx_Y_main].val/pf[idx_Y_main].p0m


        # put best value in par_ax and sort
        par_ax = np.sort(np.append(par_ax,best_val_x))
        idx_best = np.where(par_ax==best_val_x)[0][0]#get best_val position
        par_ay = np.sort(np.append(par_ay,best_val_y))
        idy_best = np.where(par_ay==best_val_y)[0][0]#get best_val position

        # create an empty array to store the likelihood
        lh = np.zeros((len(par_ax),len(par_ay),Ninteg))


        # make a 2D vector where the value of parameter ii is fixed to the values in par_ax and a Ninteg random samples are drawn randomly from the grid to set the values of the other parameters
        for i in range(len(par_ax)):
            for j in range(len(par_ay)):
                X = np.zeros((Ninteg,len(lb)))
                X[:,iiX] = par_ax[i]
                X[:,iiY] = par_ay[j]
                for k in range(len(lb)):
                    if k!=iiX and k!=iiY:
                        X[:,k] = np.random.uniform(lb[k],ub[k],Ninteg)
                lh[i,j,:] = self.LH(X,beta_scaled,N,gpr,fscale)
        
        # calculate the marginal posterior probability distribution p(w|y) for parameter ii by integrating over the other parameters
        lhlog = logsumexp(lh,axis=2) # logsumexp is more accurate than np.log(np.sum(np.exp(lh),axis=1))
        p = np.exp(lhlog-logsumexp(lhlog)) #normalize the likelihood


        # prepare the axis
        if pf[idx_X_main].optim_type == 'log':
            par_ax = 10**(par_ax)
        else:
            par_ax = par_ax * pf[idx_X_main].p0m

        if pf[idx_Y_main].optim_type == 'log':
            par_ay = 10**(par_ay)
        else:
            par_ay = par_ay * pf[idx_Y_main].p0m

        # plot the marginal posterior probability distribution p(w|y) for parameter ii
        if logscale:
            p = np.log10(p)
            vmin = np.log10(vmin)
            p[p<vmin] = vmin
            p[np.isnan(p)] = vmin
            vmax = 0
        else:
            p[p<0] = 0
            # or or p is nan
            p[np.isnan(p)] = 0
            vmin = 0
            vmax = 1

        if clear_axis: # clear the axes for the zoomed in version
            ax.clear() # clear the previous plot

        contour_=ax.contourf(par_ax,par_ay,p.T,levels=50,vmin=vmin,vmax=vmax)
        
        if pf[idx_X_main].axis_type == 'log':
            ax.set_xscale('log')
        if pf[idx_Y_main].axis_type == 'log':
            ax.set_yscale('log')
        
        if True_values is not None: # plot the true value of parameter ii if provided
            try:
                ax.plot(True_values[x_name],True_values[y_name],'C3*',markersize=10)
            except Exception as e:
                warnings.warn("At least one of the true values "+x_name+" or "+y_name+" is not provided. We skip plotting the true values.")
        
        # add best value
        ax.plot(par_ax[idx_best],par_ay[idy_best],'C2X',markersize=10)

        # plot the points
        if show_points:
            pred1_plot = [x[iiX] for x in points]
            pred2_plot = [x[iiY] for x in points]
            # convert back to the right values  
            if pf[idx_X_main].optim_type == 'log':
                pred1_plot = 10**(np.asarray(pred1_plot))
            elif pf[idx_X_main].optim_type == 'linear':
                pred1_plot = np.asarray(pred1_plot) * pf[iiX].p0m
            else:
                raise ValueError('ERROR. ',pnames[iiX],' optim_type needs to be ''linear'' or ''log'' not ',pf[iiX].optim_type,'.')

            if pf[idx_Y_main].optim_type == 'log':
                pred2_plot = 10**(np.asarray(pred2_plot))

            elif pf[idx_Y_main].optim_type == 'linear':
                pred2_plot = np.asarray(pred2_plot) * pf[iiY].p0m
            else:
                raise ValueError('ERROR. ',pnames[iiY],' optim_type needs to be ''linear'' or ''log'' not ',pf[iiY].optim_type,'.')

            ax.plot(pred1_plot,pred2_plot,'o',color='k',markersize=3)

        
        # Set the axis limits and labels
        if pf[idx_X_main].optim_type == 'log':
            ax.set_xlim([10**(lb[iiX]),10**(ub[iiX])])
        else:
            ax.set_xlim([lb[iiX]*pf[idx_X_main].p0m,ub[iiX]*pf[idx_X_main].p0m])
        ax.tick_params(axis='x', labelrotation = 45, which='both')
        if pf[idx_Y_main].optim_type == 'log':
            ax.set_ylim([10**(lb[iiY]),10**(ub[iiY])])
        else:
            ax.set_ylim([lb[iiY]*pf[idx_Y_main].p0m,ub[iiY]*pf[idx_Y_main].p0m])


        if xlabel_pos is not None:
            ax.set_xlabel(pnames_full[iiX],position = xlabel_pos)
            if xlabel_pos == 'top':
                ax.xaxis.set_label_position('top')
                ax.tick_params(axis='x',labelbottom=False,which='both')
                ax.tick_params(axis='x',labeltop=True,which='both')
            else:
                ax.xaxis.set_label_position('bottom')
                ax.tick_params(axis='x',labelbottom=True,which='both')
                ax.tick_params(axis='x',labeltop=False,which='both')

        else:
            ax.set_xticklabels([]) # remove the x label ticks  
            ax.set_xticks([]) # remove the x label ticks
            ax.set_xlabel('')
            ax.tick_params(axis='x',labelbottom=False,which='both')
            ax.tick_params(axis='x',labeltop=False,which='both')

        if ylabel_pos is not None:
            ax.set_ylabel(pnames_full[iiY],position = ylabel_pos)
            if ylabel_pos == 'right':
                ax.yaxis.set_label_position('right')
                ax.tick_params(axis='y',labelleft=False,which='both')
                ax.tick_params(axis='y',labelright=True,which='both')
            else:
                ax.yaxis.set_label_position('left')
                ax.tick_params(axis='y',labelleft=True,which='both')
                ax.tick_params(axis='y',labelright=False,which='both')
        else:
            ax.set_yticklabels([]) # remove the x label ticks  
            ax.set_yticks([]) # remove the x label ticks
            ax.set_ylabel('')
            ax.tick_params(axis='y',labelleft=False,which='both')
            ax.tick_params(axis='y',labelright=False,which='both')


        if not_init:
            ## Make colorbar
            # Define the logarithmic space for the colorbar
            cmap = plt.get_cmap('viridis')
            if logscale:
                norm = matplotlib.colors.LogNorm(vmin=10**vmin, vmax=10**vmax)
                ticks = [10**vmin,10**(int(vmin/2)),1]
            else:
                norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
                ticks = [0,0.5,1]

            # Create a scalar mappable to map the values to colors
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            # Create a colorbar
            if logscale:
                fmt = matplotlib.ticker.LogFormatterMathtext()
            else:
                fmt = None

            # add colorbar
            cbar = fig.colorbar(sm, ax=ax, ticks=ticks, format=fmt) # add the colorbar


            # Set the colorbar label on the left side
            cbar.ax.yaxis.set_label_position('right')
            # add pad for the cbar label
            # cbar.ax.set_ylabel('P('+pnames_display[iiX]+'|y)', rotation=90, va='bottom')


            
            cbar.ax.set_ylabel('P('+pnames_display[iiX]+'&'+pnames_display[iiY]+'|y)', rotation=90, va='bottom',labelpad=15)

            fig.tight_layout()
            plt.show()

    def randomize_grid_posterior(self, params, lb_main, ub_main, beta_scaled, N, gpr, fscale,kwargs_posterior, points = None, True_values = None):#, logscale = False, clear_axis = True, xlabel_pos = None, ylabel_pos = None, ax = None, fig = None, **kwargs):
        """Obtain the posterior probability distribution p(w|y) by brute force gridding
        where w is the set of model parameters and y is the data
        For each fitparameter, return mean and standard deviation


        Parameters
        ----------
        params : list of fitparameter objects
            list of fitparameters
        lb_main : list of floats
            lower bound for the fitparameters
        ub_main : list of floats
            upper bound for the fitparameters
        beta_scaled : float
            1/minimum of the scaled surrogate function
        N : integer
            number of datasets
        gpr : scikit-optimize estimator
            trained regressor
        fscale : float
            scaling factor
        kwargs_posterior : dict
            dictionary of keyword arguments for posterior function
            including: 
                Nres : integer, optional
                    Sampling resolution. Number of data points per dimension, by default 30
                Ninteg : integer, optional
                    Number of points for the marginalization over the other parameters when full_grid = False, by default 100
                full_grid : boolean, optional
                    If True, use a full grid for the posterior, by default False
                logscale : boolean, optional
                    display in log scale?, by default True
                vmin : float, optional
                    lower cutoff (in terms of exp(vmin) if logscale==True), by default 1e-100
                zoom : int, optional
                    number of time to zoom in, only used if full_grid = True, by default 0
                min_prob : float, optional
                    minimum probability to consider when zooming in we will zoom on the parameter space with a probability higher than min_prob, by default 1e-40.
                clear_axis : boolean, optional
                    clear the axis before plotting the zoomed in data, by default False.
                True_values : dict, optional
                    dictionary of true values of the parameters, by default None
                show_points : boolean, optional
                    show the explored points in the parameter space during the optimization, by default False
                savefig : boolean, optional
                    save the figure, by default False
                savefig_name : str, optional
                    name of the file to save the figure, by default 'posterior.png'
                savefig_dir : str, optional
                    directory to save the figure, by default self.res_dir
                figext : str, optional
                    extension of the figure, by default '.png'
                figsize : tuple, optional
                    size of the figure, by default (5*nb_params,5*nb_params)
                figdpi : int, optional
                    dpi of the figure, by default 300
        points : array, optional
            array of explored points in the parameter space during the optimization, by default None
        True_values : dict, optional
            dictionary of true values of the parameters, by default None
        

        Returns
        -------
        Contour plots for each pair of fit parameters
        list of float, list of float
           the mean and the square root of the second central moment 
           (generalized standard deviation for arbitrary probability distribution)
        """
        xx,stdx = [],[]
        # get varied fitparams
        pf = [pp for pp in params if pp.relRange!=0]
        pnames = [pp.name for pp in params if pp.relRange!=0]
        # p0, lb_main, ub_main = self.params_r(self.targets[-1]['params'])

        # initialize figure
        nb_params = len(pf)

        # get kwargs
        Nres = kwargs_posterior.get('Nres',10)
        Ninteg = kwargs_posterior.get('Ninteg',1e5)
        full_grid = kwargs_posterior.get('full_grid',False)
        logscale = kwargs_posterior.get('logscale',True)
        vmin = kwargs_posterior.get('vmin',1e-100)
        zoom = kwargs_posterior.get('zoom',0)
        min_prob = kwargs_posterior.get('min_prob',1e-40)
        clear_axis = kwargs_posterior.get('clear_axis',False)
        True_values = kwargs_posterior.get('True_values',None)
        show_points = kwargs_posterior.get('show_points',True)
        savefig = kwargs_posterior.get('savefig',False)
        figname = kwargs_posterior.get('figname','posterior')
        figdir = kwargs_posterior.get('figdir',self.res_dir)
        figext = kwargs_posterior.get('figext','.png')
        figsize = kwargs_posterior.get('figsize',(5*nb_params,5*nb_params))
        figdpi = kwargs_posterior.get('figdpi',300)
        show_fig = kwargs_posterior.get('show_fig',True)

        # save parameters to self for later use
        self.Nres = Nres
        self.Ninteg = Ninteg
        self.logscale = logscale
        self.vmin = vmin
        self.min_prob = min_prob


        if show_points == False:
            points = None

        fig, axes = plt.subplots(nrows=nb_params, ncols=nb_params, figsize=figsize)

        for i in range(nb_params):
            for j in range(nb_params):
                if i==j: # plot the 1D posterior on the diagonal
                    if i==0:
                        ylabel_pos = 'left'
                    else:
                        ylabel_pos = 'right'
                    if i==nb_params-1:
                        xlabel_pos = 'bottom'
                    else:
                        xlabel_pos = None
                    xmin,std,lb_new,ub_new = self.marginal_posterior_1D(pnames[i],pf=params,lb=lb_main,ub=ub_main,beta_scaled=beta_scaled,N=N,gpr=gpr,fscale=fscale,fig=fig,ax=axes[i,i],Nres=Nres,Ninteg=Ninteg,logscale=logscale,vmin=vmin,show_points=show_points,points=points,True_values=True_values,ylabel_pos=ylabel_pos,xlabel_pos=xlabel_pos)
                    xx.append(xmin)
                    stdx.append(std)
                elif i>j: # plot the 2D posterior on the lower triangle
                    if j==0:
                        ylabel_pos = 'left'
                    else:
                        ylabel_pos = None

                    if i==nb_params-1:
                        xlabel_pos = 'bottom'
                    else:
                        xlabel_pos = None

                    self.marginal_posterior_2D(pnames[j],pnames[i],pf=params,lb=lb_main,ub=ub_main,fig=fig,ax=axes[i,j],beta_scaled=beta_scaled,N=N,gpr=gpr,fscale=fscale,Nres=Nres,Ninteg=Ninteg,full_grid=full_grid,logscale=logscale,vmin=vmin,zoom=zoom,min_prob=min_prob,clear_axis=clear_axis,show_points=show_points,points=points,True_values=True_values,ylabel_pos=ylabel_pos,xlabel_pos=xlabel_pos)
                else: # plot the 2D posterior on the upper triangle
                    axes[i,j].axis('off')

        plt.tight_layout()
        if savefig:
            fig.savefig(figdir+figname+figext,dpi=figdpi)
        if show_fig:
            plt.show()
        else:
            plt.close()

        return xx,stdx,lb_new,ub_new


    def plot_objective_function(self, rrr, r, axis_type, pnames_display,kwargs_plot_obj={}):
        """Plot the objective function as a contour plot using skopt plt_objective function

        Parameters
        ----------
        rrr : _type_
            _description_
        pnames_display : _type_
            _description_
        kwargs_plot_obj : dict, optional
            kwargs for the plot_objective function, by default {}
        """            
        
        zscale = kwargs_plot_obj.get('zscale','log')
        show_fig = kwargs_plot_obj.get('show_fig',True)

         # first prepare the colormap scaling
        if zscale == 'log':
            levels = np.logspace(np.log10(rrr.func_vals.min())-0.5,np.log10(rrr.func_vals.max())+0.5,50)# levels for the contour plot, added +0.1 and -0.1 to make sure that the total objective function is plot correctly
            vmin = 10**(np.log10(rrr.func_vals.min())-0.5)
            vmax = 10**(np.log10(rrr.func_vals.max())+0.5)
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            levels = 50
            norm = None

        axo = plot_objective(rrr, n_points=50,levels =levels ,cmap='viridis',show_points=False,dimensions=pnames_display,zscale=zscale)
        axs = axo.shape

        # get kwargs
        show_points = kwargs_plot_obj.get('show_points',True)
        savefig = kwargs_plot_obj.get('savefig',False)
        figname = kwargs_plot_obj.get('figname','objective_function')
        figdir = kwargs_plot_obj.get('figdir',self.res_dir)
        figext = kwargs_plot_obj.get('figext','.png')
        figsize = kwargs_plot_obj.get('figsize',(5*axs,5*axs))
        figdpi = kwargs_plot_obj.get('figdpi',300)

        for ii in range(1,axs[0]):
            for jj in range(ii): 

                pred1_plot = [x[jj] for x in r.x_iters]
                pred2_plot = [x[ii] for x in r.x_iters]

                y = rrr.func_vals
                if show_points:
                    cset = axo[ii,jj].scatter(pred1_plot,pred2_plot,c=y.reshape(-1,),
                    edgecolor = 'black',cmap=cm.viridis,alpha=0.85,norm=norm)
                
                # Control axis ticks depending on the optim_type
                if axis_type[jj]=='log' and ii==max(range(1,axs[0])):
                    axo[ii,jj].xaxis.set_major_locator(plt.MultipleLocator(1))
                    axo[ii,jj].xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
                if axis_type[ii]=='log' and jj == 0:
                    axo[ii,jj].yaxis.set_major_locator(plt.MultipleLocator(1))
                    axo[ii,jj].yaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
                
        for ii in range(axs[0]):
            # Control axis ticks depending on the optim_type
            if axis_type[ii]=='log':
                    axo[ii,ii].xaxis.set_major_locator(plt.MultipleLocator(1))
                    axo[ii,ii].xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
                    axo[ii,ii].set_ylabel('') # remove the ylabel for the diagonal plots

        # add colorbar        
        fig = plt.gcf() # get the current figure
        cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8]) # add axes for the colorbar
        fig.colorbar(cset,cax=cb_ax , pad=10,shrink=0.8) # add the colorbar
        fig.set_size_inches(18, 16) # make it a bit larger than the original sko output
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)

        if show_fig:
            plt.show()

        return

    
    ###############################################################################
    ############################ Optimize  sko ####################################
    ###############################################################################
   
    def obj_func_sko(self, *p, params, targets, fscale, obj_type='MSE', loss='linear', threshold = 1000):
        """Objective function directly returning the loss function
        for use with the Bayesian Optimizer
        Parameters
        ----------
        *p : 1D-sequence of floats
            the parameters as passed by the Bayesian Optimizer [Larry check this]
        params : list
            list of Fitparam() objects
        model : callable
            Model function yf = f(X) to compare to y
        X : ndarray
            X Data array of size(n,m): n=number of data points, m=number of dimensions
        y : 1D-array
            data array of size (n,) to fit 
        fscale : float
            a scaling factor to keep y between 0 and 100 so the length scales can be compared
        weight : 1D-array, optional
            weight array of size (n,) to weight the data points, by default 1
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
        threshold : int, optional
            critical value above which loss sets in, by default 1000. 
            Wisely select so that the loss function affects only outliers.
            If threshold is set too low, then the value will be suppressed even fon non-outliers
            falsifying the mean square error and thus the log likelihood, the Hessian and the error bars.
        

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

            if 'threshold' in t.keys(): # if threshold is specified in targets, use it
                threshold_dum = t['threshold']
            else:
                threshold_dum = threshold

            
            if 'obj_type' in t.keys():
                obj_type_dum = t['obj_type']
            else:
                obj_type_dum = obj_type
            
            if 'collect' in self.warmstart: # WAS: 'collect in self.warmstart

                if os.path.exists(os.path.join(self.cwd,'warmstart')):
                    pass
                else:
                    os.mkdir(os.path.join(self.cwd,'warmstart'))

                filename = 'simu_target_'+str(num)
                idx = 0
                for par in params: # was enumerate(params)
                    if par.relRange != 0:
                        filename += '_' + par.name + '_{:.4e}'.format(px[idx])
                        idx += 1
                filename += '.dat'

                np.savetxt(os.path.join(self.cwd,'warmstart',filename),np.array([yf]).T)
            

            # z = np.mean(((yf-y)*weight)**2)*fscale[num]
            z = self.obj_func_metric(t,yf,obj_type=obj_type_dum)*fscale[num]

            
            if 'target_weight' in t.keys() and (isinstance(t['target_weight'], float) or isinstance(t['target_weight'], int)): # if target_weight is specified in targets, use it
                #check if target_weight is a float
                # if isinstance(t['target_weight'], float) or isinstance(t['target_weight'], int):
                    target_weights.append(t['target_weight'])
            else:
                target_weights.append(1)
                warnings.warn('target_weight for target '+str(num)+' must be a float or int. Using default value of 1.')

            zs.append(self.lossfunc(z,loss_dum,threshold=threshold_dum))

        # cost is the weigthed average of the losses
        cost = np.average(zs, weights=target_weights)    

        return cost 

    def cost_from_old_xy(self, old_xy, targets, fscale, obj_type = 'MSE', loss='linear', threshold = 1000):
        """Calculate the cost function from old data points

        Parameters
        ----------
        yfs : 1D-array
            array of size (n,) with the model function values from old data points
        y : 1D-array
            data array of size (n,) to fit
        fscale : float
            a scaling factor to keep y between 0 and 100 so the length scales can be compared
        weight : int, optional
            weight array of size (n,) to weight the data points, by default 1
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
        threshold : int, optional
            critical value above which loss sets in, by default 1000.

        Returns
        -------
        1D-array
            array of size (n,) with the loss function values
        """  
        zs = [] # list of losses
        target_weights = []  
        for num, t in enumerate(targets):
            y = t['data']['y']
            yfs = old_xy['y_'+str(num)] # get the old yf values for this target
            if 'weight' in t.keys():
                weight = t['weight']
            else:
                weight = 1

            if 'loss' in t.keys(): # if loss is specified in targets, use it
                loss_dum = t['loss']
            else:
                loss_dum = loss

            if 'threshold' in t.keys(): # if threshold is specified in targets, use it
                threshold_dum = t['threshold']
            else:
                threshold_dum = threshold

            
            if 'obj_type' in t.keys():
                obj_type_dum = t['obj_type']
            else:
                obj_type_dum = obj_type

            costs = []
            for yf in yfs:
                yf = np.asarray(yf) # needs to be an array for the obj_func_metric
                z = self.obj_func_metric(t,yf,obj_type=obj_type_dum)*fscale[num]
                # z = np.mean(((np.asarray(yf)-y)*weight)**2)*fscale[num]
                costs.append(self.lossfunc(z,loss_dum,threshold=threshold_dum))

            if 'target_weight' in t.keys(): # if target_weight is specified in targets, use it
                #check if target_weight is a float
                if isinstance(t['target_weight'], float) or isinstance(t['target_weight'], int):
                    target_weights.append(t['target_weight'])
                else:
                    target_weights.append(1)
                    warnings.warn('target_weight for target '+str(num)+' must be a float or int. Using default value of 1.')
            else:
                target_weights.append(1)
                warnings.warn('target_weight for target '+str(num)+' must be specified. Using default value of 1.')

            zs.append(costs)
        zs = np.asarray(zs).T
        # take the weighted average of the losses line by line
        costs = np.average(zs, axis=1, weights=target_weights)
        costs = list(costs)
        return costs

    def save_old_xy(self):
        """Save the old X and y values to self.SaveOldXY2file 
        """        
        path2file = self.SaveOldXY2file
        old_xy_dict = {'x':self.old_xy['x']}
        for num, t in enumerate(self.targets): # save the old y values for each target
            old_xy_dict['ydyn_'+str(num)] = self.old_xy['ydyn_'+str(num)]
            old_xy_dict['y_'+str(num)] = self.old_xy['y_'+str(num)]
        # old_xy_dict = {'x':self.old_xy['x'],'y':np.asarray(self.old_xy['y']).tolist(),'ydyn':self.old_xy['ydyn']}
        # old_xy_dict = {'x':self.old_xy['x'],'y':self.old_xy['y'],'ydyn':self.old_xy['ydyn']}
        
        with open(path2file, "w") as outfile:
            json.dump(old_xy_dict, outfile)

        return

    def load_old_xy(self):
        """Load old xy data from file from self.SaveOldXY2file
        """        
        path2file = self.Path2OldXY
        with open(path2file, "r") as outfile:
            old_xy_dict = json.load(outfile)

        self.old_xy['x'] = old_xy_dict['x']
        # self.old_xy['y'] = old_xy_dict['y']
        # self.old_xy['ydyn'] = old_xy_dict['ydyn']
        for num, t in enumerate(self.targets): # save the old y values for each target
            self.old_xy['y_'+str(num)] = old_xy_dict['y_'+str(num)]
            self.old_xy['ydyn_'+str(num)] = old_xy_dict['ydyn_'+str(num)]

        return

    def optimize_sko_parallel(self,n_jobs=4,n_yscale=20, n_BO=10, n_initial_points = 10,n_BO_warmstart=5,n_jobs_init=None,obj_type='MSE',loss='linear',threshold=1000,kwargs=None, base_estimator = 'GP',show_objective_func=True,kwargs_plot_obj=None,show_posterior=True,kwargs_posterior=None,verbose=True):
        """Multi-objective optimization of the parameters of the model using the scikit-optimize package

        Parameters
        ----------
        n_jobs : int, optional
            number of parallel jobs to run, by default 4
        n_yscale : int, optional
            number of points used to estimate the scaling factor yscale, by default 20
        n_BO : int, optional
            number of points to run in the Bayesian optimization, by default 10
        n_initial_points : int, optional
            number of initial points to run, by default 10
        n_BO_warmstart : int, optional
            number of points to run in the Bayesian optimization after warmstart, by default 5
        n_jobs_init : int, optional
            number of parallel jobs to run for the initial points, by default None
            if None, then n_jobs is used
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
        threshold : int, optional
            critical value above which loss sets in, by default 1000. 
            Wisely select so that the loss function affects only outliers.
            If threshold is set too low, then the value will be suppressed even fon non-outliers
            falsifying the mean square error and thus the log likelihood, the Hessian and the error bars.
        suggest_only : bool, optional
            only suggest the next point and does not evaluate it, by default False
        kwargs : dict, optional
            dictionary of keyword argument to check for the improvement of the model, by default None
            including:

                max_loop_no_improvement : int, optional
                    maximum number of loops without improvement, by default 10
                check_improvement : bool, optional
                    check for improvement can be either None, 'relax', 'strict', by default None
                    if None, then no check is performed
                    if 'relax', then the check is performed by checking if abs(fun_new - fun_best)/fun_new > ftol or if norm(dx) < xtol*(xtol + norm(x))
                    if 'strict', then the check is performed by checking if abs(fun_new - fun_best)/fun_new > ftol only
                ftol : float, optional
                    monitor the change in the minimum of the objective function value, by default 1e-3
                xtol : float, optional
                    Monitor the change of the fitting results, by default 1e-3
                initial_point_generator : str, optional
                    type of initial point generator, can be ['random','sobol','halton','hammersly','lhs','grid'], by default 'lhs'
                acq_func : str, optional
                    type of acquisition function, can be ['LCB','EI','PI','gp_hedge'], by default 'gp_hedge'
                acq_optimizer : str, optional
                    type of acquisition optimizer, can be ['auto','sampling','lbfgs'], by default 'auto'
                acq_func_kwargs : dict, optional
                    dictionary of keyword arguments for the acquisition function, by default {}
                acq_optimizer_kwargs : dict, optional
                    dictionary of keyword arguments for the acquisition optimizer, by default {}
                switch2exploit : bool, optional
                    switch to exploitation after reaching max_loop_no_improvement loops without improvement and reset the counter, by default True
                    
        
        show_objective_func : bool, optional
            plot the objective function, by default True
        kwargs_plot_obj : dict, optional
            dictionary of keyword arguments for plot_objective_function, by default None
            including:

                zscale: str, optional
                    type of scaling to use for the objective function, can be ['linear','log'], by default 'log'
                show_points : boolean, optional
                    show the explored points in the parameter space during the optimization, by default False
                savefig : boolean, optional
                    save the figure, by default False
                savefig_name : str, optional
                    name of the file to save the figure, by default 'posterior.png'
                savefig_dir : str, optional
                    directory to save the figure, by default self.res_dir
                figext : str, optional
                    extension of the figure, by default '.png'
                figsize : tuple, optional
                    size of the figure, by default (5*nb_params,5*nb_params)
                figdpi : int, optional
                    dpi of the figure, by default 300

        show_posterior : bool, optional
            calculate & show posterior distribution, by default True
        kwargs_posterior : dict
            dictionary of keyword arguments for posterior function, by default None
            including: 
            
                Nres : integer, optional
                    Sampling resolution. Number of data points per dimension, by default 30
                Ninteg : integer, optional
                    Number of points for the marginalization over the other parameters when full_grid = False, by default 100
                full_grid : boolean, optional
                    If True, use a full grid for the posterior, by default False
                randomize : boolean, optional
                    If True, calculate the posterior for all the dimension but draw the marginalization points randomly and do not use a corse grid, by default False
                logscale : boolean, optional
                    display in log scale?, by default True
                vmin : float, optional
                    lower cutoff (in terms of exp(vmin) if logscale==True), by default 1e-100
                zoom : int, optional
                    number of time to zoom in, only used if full_grid = True, by default 0
                min_prob : float, optional
                    minimum probability to consider when zooming in we will zoom on the parameter space with a probability higher than min_prob, by default 1e-40.
                clear_axis : boolean, optional
                    clear the axis before plotting the zoomed in data, by default False.
                True_values : dict, optional
                    dictionary of true values of the parameters, by default None
                show_points : boolean, optional
                    show the explored points in the parameter space during the optimization, by default False
                savefig : boolean, optional
                    save the figure, by default False
                savefig_name : str, optional
                    name of the file to save the figure, by default 'posterior.png'
                savefig_dir : str, optional
                    directory to save the figure, by default self.res_dir
                figext : str, optional
                    extension of the figure, by default '.png'
                figsize : tuple, optional
                    size of the figure, by default (5*nb_params,5*nb_params)
                figdpi : int, optional
                    dpi of the figure, by default 300
        verbose : bool, optional
            display progress and results, by default True

        Returns
        -------
        dict
            dictionary with the optimized parameters ('popt') and the corresponding covariance ('pcov') and standard deviation ('std') values
        """  

        self.verbose = verbose
        if kwargs is None:
            kwargs = {}
        max_loop_no_improvement = kwargs.get('max_loop_no_improvement',10)
        check_improvement = kwargs.get('check_improvement','relax')
        xtol = kwargs.get('xtol',1e-3)
        ftol = kwargs.get('ftol',1e-3)
        initial_point_generator = kwargs.get('initial_point_generator','lhs')
        acq_func = kwargs.get('acq_func','gp_hedge')
        acq_optimizer = kwargs.get('acq_optimizer','auto')
        acq_func_kwargs = kwargs.get('acq_func_kwargs',{})
        acq_optimizer_kwargs = kwargs.get('acq_optimizer_kwargs',{})
        switch2exploit = kwargs.get('switch2exploit',True)

        if check_improvement not in [None,'relax','strict']:
            check_improvement = None
            warnings.warn("check_improvement should be either None, 'relax' or 'strict'. Setting it to 'None")
        if max_loop_no_improvement:
            if type(max_loop_no_improvement) is not int:
                warnings.warn("max_loop_no_improvement should be an integer. Setting it to 10")
                max_loop_no_improvement = 10

        if n_jobs_init is None: # if n_jobs_init is not specified, use n_jobs
            n_jobs_init = n_jobs
        
        n_jobs_init = min(n_jobs_init,n_initial_points) # make sure that we don't run more jobs than initial points
        # check that n_jobs_init is smaller than n_initial_points otherwise optimizer.ask() will fail
        if n_jobs_init > n_initial_points:
            n_jobs_init = int(n_initial_points/2) # make sure that we don't run more jobs than initial points
            warnings.warn("n_jobs_init is larger than n_initial_points. Setting n_jobs_init to n_initial_points/2 = "+str(n_jobs_init))
            
        # prepare self.old_xy
        self.old_xy = {'x':[]} # dict to hold the calculations from a previous run if warmstart==True
        for num,t in enumerate(self.targets):
            self.old_xy['y_'+str(num)] = []
            self.old_xy['ydyn_'+str(num)] = 1

        # Make sure that we have something to recall if needed
        if (self.Path2OldXY is not None) and ('recall' in self.warmstart):
            self.load_old_xy()
            print('Old_xy data loaded from '+self.Path2OldXY)


        # check that we don't have any conflict in the warmstart option
        if self.warmstart is not None:
            if 'collect_init' in self.warmstart and 'collect_BO' in self.warmstart:
                self.warmstart = 'collect'
                warnings.warn('warmstart = "collect_init" and "collect_BO" are mutually exclusive. Setting warmstart = "collect"')

            if  'recall' in self.warmstart:
                if (len(self.old_xy['x']) == 0):
                    print(len(self.old_xy['x']))
                    raise ValueError('No old_xy data found. Please run the optimization once before using the warmstart option. len'+str(len(self.old_xy['x'])))
            
        
        gprs = [] # container for the trained gprs of all targets   
    
        p0,lb,ub = self.params_r(self.params) # read out Fitparams & respect settings

        pnames = [pp.name for pp in self.params if pp.relRange!=0]
        pnames_display = [pp.display_name for pp in self.params if pp.relRange!=0] # for plots axis labels
        axis_type = [pp.optim_type for pp in self.params if pp.relRange!=0] # to get the type of the axis for the plots

        # dimensions = [Real(up,lo) for up,lo in zip(lb,ub)] # sampling space
        dimensions = [] # sampling space
        idx = 0
        got_categorical = False
        for num,pp in enumerate(self.params):
            if pp.relRange != 0:
                if pp.val_type == 'float':
                    dimensions.append(Real(lb[idx],ub[idx]))
                    idx += 1
                elif pp.val_type == 'int':
                    dimensions.append(Integer(lb[idx],ub[idx]))
                    idx += 1
                elif pp.val_type == 'str':
                    dimensions.append(Categorical(pp.lims))
                    got_categorical = True
                else:
                    raise ValueError('val_type should be either "float", "int" or "str"')

        if got_categorical:
            if check_improvement is not None and check_improvement != 'strict':
                check_improvement = 'strict' # if we have categorical variables, we need to check strictly for improvement
                warnings.warn('check_improvement is set to "strict" because we have categorical variables')
            if show_posterior == True:
                show_posterior = False
                warnings.warn('show_posterior is set to False because we have categorical variables')

        ## First do a blind optimization to get an idea of the total y dynamics which lets us calculate a scaling factor
        if 'recall' not in self.warmstart: 

            # run the target one by one to get the individual ydyn
            fscales = []
            for num, t in enumerate(self.targets):            
                pfunc = partial(self.obj_func_sko,params = self.params,targets=[t],fscale=[1],obj_type=obj_type,loss=loss,threshold=threshold)
                
                if self.verbose:
                    print('finding scaling factor for y')
                
                optimizer = Optimizer(dimensions, base_estimator = base_estimator, n_random_starts = None,
                            n_initial_points = n_initial_points,initial_point_generator=initial_point_generator,n_jobs=n_jobs_init,
                            acq_func = acq_func, acq_optimizer = acq_optimizer,random_state = None,model_queue_size=None,acq_func_kwargs=acq_func_kwargs,acq_optimizer_kwargs=acq_optimizer_kwargs)

                if not isinstance(base_estimator,str): #is either a string or a skl estimator
                    optimizer.base_estimator_= base_estimator

                nloop = int(np.ceil(n_yscale/n_jobs_init))

                
                for ii in tnrange(nloop,desc='Scaling runs for target '+str(num)):
                    x = optimizer.ask(n_points=n_jobs_init)
                    if self.parallel:
                        y = Parallel(n_jobs=n_jobs)(delayed(pfunc)(v) for v in x)
                    else:
                        y = [pfunc(v) for v in x]
                    r0 = optimizer.tell(x,y)

                
                ydyn = np.max(r0.func_vals) - np.min(r0.func_vals)
                fscales.append(1/ydyn*100)
                if 'collect' in self.warmstart:
                    self.old_xy['ydyn_'+str(num)] = ydyn

                if self.verbose:
                    print(f'ydyn_'+str(num)+' = {ydyn}')
                    print(f'The scaling factor is set at fscale = {1/ydyn*100}')

        else:
            
            fscales = []
            for num, t in enumerate(self.targets):
                ydyn = self.old_xy['ydyn_'+str(num)]
                fscales.append(1/ydyn*100)

            if self.verbose:
                print('Warmstart option: re-using old ydyn:', ydyn)

        ## Now do the actual optimization
        # with y between 0 and 100 (to make the sko Matern kernel happy) 

        pfunc = partial(self.obj_func_sko,params = self.params,targets=self.targets,fscale=fscales,obj_type=obj_type,loss=loss,threshold=threshold)
        # run the optimization 
        optimizer = Optimizer(dimensions, base_estimator = base_estimator, n_random_starts = None,
                    n_initial_points = n_initial_points,initial_point_generator=initial_point_generator,n_jobs=n_jobs_init,
                    acq_func = acq_func, acq_optimizer = acq_optimizer,random_state = None,model_queue_size=None,acq_func_kwargs=acq_func_kwargs,acq_optimizer_kwargs=acq_optimizer_kwargs)

        if not isinstance(base_estimator,str): #is either a string or a skl estimator
            optimizer.base_estimator_= base_estimator
        
        
        # Start with the initial points
        # if verbose:
        tic = time()
        print('Starting with initial points')
        check_length = True
        if ('recall' in self.warmstart): # check if we have old_xy as the right length
            for num,t in enumerate(self.targets):
                if len(t['data']['y']) != np.asarray(self.old_xy['y_'+str(num)]).shape[1]:
                    check_length = False
                    print('/!\ Warmstart option could not be used: old_xy data does not match current data. Starting from scratch. /!\ ')
                    print('Next time ensure that the old_xy lenght and input data matches the data to fit.')


        if ('recall' not in self.warmstart) or ('recall' in self.warmstart and not check_length):
            
            nloop_init = int(np.ceil(n_initial_points/n_jobs_init))
            nloop_BO = int(np.ceil(n_BO/n_jobs))
            
            for ii in tnrange(nloop_init,desc='Initial points'):
                x = optimizer.ask(n_points=n_jobs_init)
                if self.parallel:
                    y = Parallel(n_jobs=n_jobs_init)(delayed(pfunc)(v) for v in x)
                else:
                    y = [pfunc(v) for v in x]

                if ('collect_init' in self.warmstart) or (('collect' in self.warmstart) and 'collect_BO' not in self.warmstart): # collect the initial points
                    self.old_xy['x'].extend(x)
                    for num,t in enumerate(self.targets):
                        # self.old_xy['x'].extend(x)
                        for old in x:
                            filename = 'simu_target_'+str(num)
                            idx = 0
                            for par in self.params:
                                if par.relRange != 0:
                                    filename += '_' + par.name + '_{:.4e}'.format(old[idx])
                                    idx += 1
                            filename += '.dat'
                            oldy = np.loadtxt(os.path.join(self.cwd,'warmstart',filename))
                            self.old_xy['y_'+str(num)].append(list(oldy))
                
                rrr = optimizer.tell(x,y)
            
                if len(rrr.models)>0:
                    xmin,funmin = expected_minimum(rrr)
                    if self.verbose:
                        print('Minimum of surrogate function:',xmin,'with function value',funmin)
            
        else:
            print('Warmstart; using old_xy data')
            y = self.cost_from_old_xy(self.old_xy,self.targets,fscale=fscales,obj_type=obj_type,loss=loss,threshold=threshold) # get cost functions from old simulations, but with new data
            print(len(self.old_xy['x']))
            rrr = optimizer.tell(self.old_xy['x'],y,fit=True) # send result to optimizer

            nloop_BO = int(np.ceil((n_BO_warmstart)/n_jobs)) # reduce number of necessary new evaluations accordingly
            if self.verbose:
                print('Warmstart; reducing number of new evaluations to', nloop_BO*n_jobs)

        print('Initial points done in {:.2f} s'.format(time()-tic))
        # Now do the actual BO
        # if verbose:
        print('Starting with BO')
        tic = time()
        
        no_improvement_loop = 1
        best_funmin = np.inf
        best_xmin = None
        
        if 'collect_BO' in self.warmstart:
            # clear old_xy to avoid saving old data too
            for num,t in enumerate(self.targets):
                self.old_xy['y_'+str(num)] = []
                self.old_xy['x'] = []

        for ii in tnrange(nloop_BO,desc='BO runs'):
            x = optimizer.ask(n_points=n_jobs)
            if self.parallel:
                y = Parallel(n_jobs=n_jobs)(delayed(pfunc)(v) for v in x)
            else:
                y = [pfunc(v) for v in x]
                
            if ('collect' in self.warmstart and 'collect_init' not in self.warmstart) or 'collect_BO' in self.warmstart: 
                self.old_xy['x'].extend(x)
                for num,t in enumerate(self.targets):
                    # self.old_xy['x'].extend(x)
                    for old in x:
                        filename = 'simu_target_'+str(num)
                        idx = 0
                        for par in self.params:
                            if par.relRange != 0:
                                filename += '_' + par.name + '_{:.4e}'.format(old[idx])
                                idx += 1
                        filename += '.dat'
                        oldy = np.loadtxt(os.path.join(self.cwd,'warmstart',filename))
                        self.old_xy['y_'+str(num)].append(list(oldy))
            
            rrr = optimizer.tell(x,y)
            
            if len(rrr.models)>0:
                xmin,funmin = expected_minimum(rrr)
                if self.verbose:
                    print('Minimum of surrogate function:',xmin,'with function value',funmin)

                if check_improvement is not None and max_loop_no_improvement > 0 and xtol is not None and ftol is not None:
                    imin = np.argmin(optimizer.yi)
                    funmin_ground = optimizer.yi[imin]
                    xmin_ground = optimizer.Xi[imin]
                    
                    
                    if check_improvement == 'strict':
                        if abs(funmin_ground - best_funmin)/funmin_ground > ftol:
                            best_funmin = funmin_ground
                            best_xmin = xmin
                            no_improvement_loop = 1
                            if self.verbose:
                                print('Still improving...')
                        else:
                            if self.verbose:
                                print('No improvement in {} consecutive loops'.format(no_improvement_loop))
                            no_improvement_loop += 1
                            if no_improvement_loop >= max_loop_no_improvement and switch2exploit:
                                if self.verbose:
                                    print('Switching to exploitation')
                                optimizer.acq_func = 'LCB'
                                optimizer.acq_func_kwargs = {"xi": 0.000001, "kappa": 0.001}
                                switch2exploit = False
                                no_improvement_loop = 1
                                optimizer.update_next()
                            elif no_improvement_loop >= max_loop_no_improvement and not switch2exploit:
                                if self.verbose:
                                    print('No improvement in {} consecutive loops, we are stopping the BO early'.format(max_loop_no_improvement))
                                break
                    elif check_improvement == 'relax':                            
                        if abs(funmin_ground - best_funmin)/funmin_ground > ftol:
                            best_funmin = funmin_ground
                            best_xmin = xmin
                            no_improvement_loop = 1
                            if self.verbose:
                                print('Still improving...')
                        elif abs(funmin_ground - best_funmin)/funmin_ground <= ftol and np.linalg.norm(np.asarray(xmin_ground) - np.asarray(best_xmin)) > xtol*(xtol+np.linalg.norm(np.asarray(best_xmin))):
                            no_improvement_loop = 1
                            best_xmin = xmin_ground
                            if self.verbose:
                                print('Still evolving...')
                        else:
                            if self.verbose:
                                print('No improvement in {} consecutive loops'.format(no_improvement_loop))
                            no_improvement_loop += 1
                            if no_improvement_loop >= max_loop_no_improvement and switch2exploit:
                                if self.verbose:
                                    print('Switching to exploitation')
                                optimizer.acq_func = 'LCB'
                                optimizer.acq_func_kwargs = {"xi": 0.000001, "kappa": 0.001}
                                switch2exploit = False
                                no_improvement_loop = 1
                                optimizer.update_next()
                            elif no_improvement_loop >= max_loop_no_improvement and not switch2exploit:
                                if self.verbose:
                                    print('No improvement in {} consecutive loops, we are stopping the BO early'.format(max_loop_no_improvement))
                                break
                            
                    else:
                        pass

        print('BO done in {:.2f} s'.format(time()-tic))


        imin = np.argmin(optimizer.yi)
        r = SimpleNamespace()
        r.fun = optimizer.yi[imin]
        r.x = optimizer.Xi[imin]  
        r.space = optimizer.space
        r.x_iters = rrr.x_iters
        
        ## Optimization finished 
        print('Ground truth minimum at:', r.x, 'with function value:',r.fun)

        # plot the objective function
        if show_objective_func:
            self.plot_objective_function(rrr, r, axis_type, pnames_display, kwargs_plot_obj = kwargs_plot_obj)    

        gpr = deepcopy(rrr.models[-1]) # take the last of the models (for some reason it is not trained)
        if not got_categorical: # if there are categorical variables, the gpr is not trained (need to figure this out later)
            gpr.fit(rrr.x_iters,rrr.func_vals) # train it
        gprs.append(deepcopy(gpr))

        # save the old_xy dictionary to file
        if 'collect' in self.warmstart and self.SaveOldXY2file is None:
            print('Warning: you are collecting the data but not saving it to file. This is not recommended. Set SaveOldXY2file to a filename to save the data to file.')

        if 'collect' in self.warmstart and self.SaveOldXY2file is not None:
            self.save_old_xy() # save the old_xy dictionary to file

        if len(rrr.models)>0:
            xmin,funmin = expected_minimum(rrr)
            print('Minimum of surrogate function:',xmin,'with function value',funmin)


        # calculate beta
        # Christopher M. Bishop:Pattern Recognition and Machine Learning, Springer Information Science & statistics
        # Chapter 1.2.5 pg 29 eq- 1.63
        if funmin>0:
            beta = 1/funmin # precision = 1/sigma**2 !!! rr.fun is MSE which is OK because consided below in LLH()!! 
        else:
            beta = abs(1/funmin) # precision = 1/sigma**2 !!! rr.fun is MSE which is OK because consided below in LLH()!!
            #Add warning here
            warnings.warn('The surrogate function got negative. setting beta to the absolute value of the negative value')
        

        # number of data points
        Num_data_pts = 0
        for num,t in enumerate(self.targets): # get the number of data points
            Num_data_pts += len(t['data']['y'])
            # if type(t['data']['X']) is list: # safety check for the case when the data is a list and not an np arrays
            #     Num_data_pts += np.array(t['data']['X']).shape[0]
            # else:
            #     Num_data_pts += t['data']['X'].shape[0]

        # save parameters to self for later use
        self.N = Num_data_pts
        self.gpr = gpr
        self.fscale = fscales
        self.beta_scaled = beta
        self.points = r.x_iters
        self.kwargs_posterior = kwargs_posterior

        self.params_w(r.x,self.params) # write the best fit parameters to self.params
        # get the posterior probabiliy distribution p(w|t)
        if show_posterior:
            
            pf = [pp for pp in self.params if pp.relRange!=0]
            p0, lb_main, ub_main = self.params_r(self.params)
            self.points = r.x_iters
            xmin0,std = self.posterior(pf, lb_main, ub_main,points=r.x_iters,beta_scaled = beta,N=Num_data_pts,gpr = gpr,fscale=fscales,kwargs_posterior=kwargs_posterior)

            # Note: the posterior is calculated with from the surrogate function and not the ground truth function therefore it is not always accurate
            #       especially when the surrogate function is not trained well. This is why the best fit parameters are taken from the best one sampled by the BO
            #       and not from the posterior. The posterior is only used to get the error bars. This mean that sometimes the best fit parameters is not necessarily
            #       the one with the highest posterior probability. In this case the 95% confidence interval that is outputted is stretched to include the best fit parameters.
            #       This is not a problem because the posterior is not used to get the best fit parameters but only to get the error bars, this way guarantee that the best fit is 
            #       always within the outputted error bars.
            #       The 95% interval is stored in std (so std is NOT the standard deviation of the posterior distribution)
        
            self.params_w(r.x,self.params,std=std) # read out Fitparams & respect settings


        return {'popt':xmin,'r':rrr,'GrMin':r.x}
        
    
    # ###############################################################################
    # ################################# Curve fit ###################################
    # ###############################################################################

    # def obj_func_curvefit(self,X,*p,params,model):
    #     """Objective function as desired by scipy.curve_fit

    #     Parameters
    #     ----------
    #     X : ndarray
    #         X Data array of size(n,m): n=number of data points, m=number of dimensions
    #     *p : ndarray
    #         list of float; values of the fit parameters as supplied by the optimizer
    #     params : list
    #         list of Fitparam() objects
    #     model : callable
    #         Model function yf = f(X) to compare to y

    #     Returns
    #     -------
    #     1D-array
    #         array of size (n,) with the model values
    #     """ 

    #     yfit = []
    #     for t in self.targets:
    #         xdata = t['data']['X'] # get array of experimental X values
    #         ydata = t['data']['y'] # get vector of experimental y values


    #         self.params_w(p, params) # write variable parameters into params, respecting user settings
    #         y = list(t['model'](xdata,params))
    #         yfit = yfit + y

    #     yfit = np.array(yfit)
    #     #print('errsq', np.sum(((yfit-self.yfull)/self.weightfull))**2)
    #     return yfit

    # def optimize_curvefit(self, kwargs = None):
    #     """Use curvefit to optimize a function y = f(x) where x can be multi-dimensional
    #     use this function if the loss function is deterministic
    #     do not use this function if loss function has uncertainty (e.g from numerical simulation)
    #     in this case, use optimize_sko

    #     Parameters
    #     ----------
    #     kwargs : dict, optional
    #         kwargs aruguments for curve_fit, see scipy.optimize.curve_fit documentation for more information, by default None\\
    #         If no kwargs is provided use:\\
    #         kwargs = {'ftol':1e-8, 'xtol':1e-6, 'gtol': 1e-8, 'diff_step':0.001,'loss':'linear','max_nfev':5000}

    #     Returns
    #     -------
    #     dict
    #         dictionary with the optimized parameters ('popt') and the corresponding covariance ('pcov') and standard deviation ('std') values
    #     """  
    #     xfull = []
    #     yfull = []
    #     weightfull = []

    #     for t in self.targets:
    #         xdata = t['data']['X'] # get array of experimental X values
    #         ydata = list(t['data']['y']) # get vector of experimental y values
    #         if len(xfull)==0:
    #             xfull = xdata
    #             yfull = ydata 
    #             weightfull = list(1/t['weight'])
    #         else:
    #             xfull = np.vstack((xfull,xdata))
    #             yfull = yfull + ydata
    #             weight = list(1/t['weight'])
    #             weightfull = weightfull + weight
                
    #         if kwargs == None:
    #             kwargs = {'ftol':1e-8, 'xtol':1e-6, 'gtol': 1e-8, 'diff_step':0.001,'loss':'linear','max_nfev':5000}


    #     p0,lb,ub = self.params_r(self.params) # read out Fitparams & respect settings    
    #     # using partial functions allows passing extra parameters
    #     pfunc = partial(self.obj_func_curvefit, params = self.params, model = t['model'])

    #     r = curve_fit(pfunc,xfull,yfull,p0=p0,sigma=weightfull,absolute_sigma=False,check_finite=True,bounds=(lb,ub),full_output=True,method=None,jac=None, **kwargs)
    #     popt = r[0]
    #     pcov = r[1]
    #     infodict = r[2]
    #     mesg = r[3]
    #     std = np.sqrt(np.diag(pcov))
    #     # make tuple with (std,std) for each parameter
    #     stdx = []
    #     for s in std:
    #         stdx.append((s,s))
    #     self.params_w(popt,self.params,std=stdx) # write variable parameters into params, respecting user settings

    #     return {'popt':popt,'pcov':pcov,'std':std,'infodict':infodict}

    
    def single_point(self,X,y,params,n_jobs=4,base_estimator='GP',n_initial_points = 100,show_objective_func=True,kwargs_plot_obj=None,axis_type=[],show_posterior=True,kwargs_posterior=None):
        """Do a single Gaussian Process Regression on the X,y data

        Parameters
        ----------
        X : ndarray
            X Data array of size(n,m): n=number of data points, m=number of dimensions
        y : 1D-array
            data array of size (n,) to fit 
        params : list
            list of Fitparam() objects
        base_estimator : str, optional
            base estimator for the Gaussian Process Regression, by default 'GP'
        n_initial_points : int, optional
            number of initial points to use for the Gaussian Process Regression, by default 100
        show_objective_func : bool, optional
            whether to plot the objective function or not, by default True
        kwargs_plot_obj : dict, optional
            kwargs arguments for the plot_objective_function function, by default None
        axis_type : list, optional
            list of strings with the type of axis to use for each dimension. Either 'linear' or 'log', by default []
        verbose : bool, optional
            whether to display progress and results or not, by default True
        zscale : str, optional
            scale to use for the z axis of the contour plots. Either 'linear' or 'log', by default 'linear'
        show_plots : bool, optional
            whether to show the plots or not, by default True
        """        
        zscale = kwargs_plot_obj.get('zscale','log')

        # if axis_type == []:
        #     axis_type = [zscale for i in range(len(X[0]))]
        axis_type = [pp.axis_type for pp in params if pp.relRange!=0] # to get the type of the axis for the plots
        

        # do a single gpr on the X,y data
        Xa = np.array(X)
        dimensions = [Real(np.min(Xa[:,ii]), np.max(Xa[:,ii])) for ii in range(Xa.shape[1])] # sampling space
        optimizer = Optimizer(dimensions, base_estimator = base_estimator, n_random_starts = None,
                    n_initial_points = n_initial_points,initial_point_generator='lhs',n_jobs=n_jobs,
                    acq_func = 'gp_hedge', acq_optimizer = 'auto',random_state = None,model_queue_size=None)
        rrr = optimizer.tell(X,y)
        imin = np.argmin(optimizer.yi)
        r = SimpleNamespace()
        r.fun = optimizer.yi[imin]
        r.x = optimizer.Xi[imin]  
        r.space = optimizer.space
        r.x_iters = rrr.x_iters

        if self.verbose:
            print('Ground truth minimum at:', r.x, 'with function value:',r.fun)

        # plot the objective function
        if show_objective_func:
            pnames_display = [pp.display_name for pp in params if pp.relRange!=0] # for plots axis labels
            self.plot_objective_function(rrr, r, axis_type, pnames_display, kwargs_plot_obj = kwargs_plot_obj)   


        gpr = deepcopy(rrr.models[-1]) # take the last of the models (for some reason it is not trained)
        gpr.fit(rrr.x_iters,rrr.func_vals) # train it
        
        self.gpr_single = gpr # save it for later use
        self.params_single = params # save it for later use

        if len(rrr.models)>0:
            xmin,funmin = expected_minimum(rrr)
            if self.verbose:
                print('Minimum of surrogate function:',xmin,'with function value',funmin)
        # Christopher M. Bishop:Pattern Recognition and Machine Learning, Springer Information Science & statistics
        # Chapter 1.2.5 pg 29 eq- 1.63
        if funmin>0:
            beta = 1/funmin # precision = 1/sigma**2 !!! rr.fun is MSE which is OK because consided below in LLH()!! 
        else:
            beta = abs(1/funmin) # precision = 1/sigma**2 !!! rr.fun is MSE which is OK because consided below in LLH()!!
            #Add warning here
            warnings.warn('The surrogate function got negative. setting beta to the absolute value of the negative value')

        self.params_w(xmin,params) # write variable parameters into params, respecting user settings
        # Number of data points
        Num_data_pts = 0
        for num,t in enumerate(self.targets): # get the number of data points
            if type(t['data']['X']) is list: # safety check for the case when the data is a list and not an np arrays
                Num_data_pts += np.array(t['data']['X']).shape[0]
            else:
                Num_data_pts += t['data']['X'].shape[0]
        
        # if self.N not an atritube of the class, then set it to the number of data points
        if not hasattr(self,'N'):
            self.N = Num_data_pts
        # get the posterior probabiliy distribution p(w|t)
        if show_posterior:
            # params = [pp for pp in self.targets[-1]['params'] if pp.relRange!=0]
            p0, lb_main, ub_main = self.params_r(params)
            self.points_fom = r.x_iters
            xmin0,std = self.posterior(params,lb_main, ub_main,points=r.x_iters,beta_scaled = beta,N=Num_data_pts,gpr = gpr,fscale=[1],kwargs_posterior=kwargs_posterior)
            # std from the posterior distribution should be better than those from the Hessian
            # but we discard xmin0 because we rely on the best fit not the surrogate
            self.params_w(xmin,params,std=std) # write variable parameters into params, respecting user settings
        
        self.params_single = params # save the foms for later use
        self.points_single = r.x_iters # save the foms points for later use
        if show_posterior: 
            return xmin,std
        else:
            return xmin

