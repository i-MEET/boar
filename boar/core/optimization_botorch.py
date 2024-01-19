######################################################################
################ MultiObjectiveOptimizer class #######################
######################################################################
# Authors: 
# Larry Lueer (https://github.com/larryluer)
# Vincent M. Le Corre (https://github.com/VMLC-PV)
# (c) i-MEET, University of Erlangen-Nuremberg, 2021-2022-2023 

# Import libraries
from __future__ import annotations
from abc import ABC, abstractmethod

import os,json,copy,warnings,matplotlib,itertools
import numpy as np
import pandas as pd
from time import time
from functools import partial
from matplotlib import pyplot as plt
from matplotlib import cm
from copy import deepcopy
from tqdm import tnrange, tqdm_notebook
import seaborn as sns
from joblib import Parallel, delayed
from scipy.optimize import minimize as sp_minimize
from scipy.special import logsumexp
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
# sns.set_theme(style="ticks")
# import ray
# Import BOtorch and Ax libraries
import torch
from ax import *
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax import Data, Experiment, ParameterType, RangeParameter, SearchSpace
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.core.metric import Metric
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df
from ax.core.observation import ObservationFeatures
# from ax.runners.synthetic import SyntheticRunner
# from ax.modelbridge.factory import get_MOO_EHVI, get_MOO_PAREGO, get_MOO_NEHVI # Factory methods for creating multi-objective optimization modesl.

# Analysis utilities, including a method to evaluate hypervolumes
from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax.modelbridge.torch import infer_objective_thresholds



# Import boar libraries
from boar.core.optimizer import BoarOptimizer # import the base class


class MooBOtorch(BoarOptimizer):
    # a class for multi-objective optimization
    def __init__(self,params = None, targets = None, parameter_constraints = None, warmstart = None, Path2OldXY = None, SaveOldXY2file = None, res_dir = 'temp', evaluate_custom = None, parallel = True, verbose = False) -> None:
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
            directory where the results will be saved, by default 'temp'
        evaluate_custom : function, optional
            use a custom evaluation function instead of the default one, this is useful when the same model is used for different targets, by default None
        parallel : bool, optional
            use parallelization, if False n_jobs can still be > 1 but the evaluation will be done sequentially, by default True
        verbose : bool, optional
            print some information, by default False

        """        
        if targets != None:
            self.targets = targets
        
        if params != None:
            self.params = params
        
        if parameter_constraints != None:
            self.parameter_constraints = parameter_constraints
        else:
            self.parameter_constraints = None
        
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

        self.evaluate_custom = evaluate_custom
        self.parallel = parallel

    
    
    def ConvertParams(self, params):
        """Convert the params to the format required by the Ax/Botorch library

        Parameters
        ----------
        params : list of Fitparam() objects
            list of Fitparam() objects

        Returns
        -------
        list of dict
            list of dictionaries with the following keys:\\
                'name': string: the name of the parameter\\
                'type': string: 'range' or 'fixed'\\
                'bounds': list of float: the lower and upper bounds of the parameter\\
                
        """        
        Axparams = []
        for par in params:
            if par.relRange != 0:
                if par.val_type == 'float':
                    if par.optim_type == 'log':
                        bounds = [np.log10(par.lims[0]), np.log10(par.lims[1])]
                    else:
                        bounds = [par.lims[0]/par.p0m, par.lims[1]/par.p0m] # remove the order of magnitude 

                    Axparams.append({'name':par.name, 'type':'range', 'bounds':bounds, 'value_type':'float'})
                elif par.val_type == 'int':
                    bounds = [round(par.lims[0]/par.p0m),round(par.lims[1]/par.p0m)]
                    Axparams.append({'name':par.name, 'type':'range', 'bounds':bounds, 'value_type':'int'})
                    
                elif par.val_type == 'str':
                    bounds = par.lims
                    Axparams.append({'name':par.name, 'type':'choice', 'values':bounds, 'value_type':'str'})
                else:
                    raise ValueError('val_type must be either float, int or str')

        return Axparams
    
    
    def get_model(self,estimator='GPEI',use_CUDA=True):
        """Get the model

        Parameters
        ----------
        estimator : str, optional
            Estimator to use. The default is 'GPEI'.
        use_CUDA : bool, optional
            Use CUDA. The default is True.

        Raises
        ------
        ValueError
            If the estimator is not implemented yet.

        Returns
        -------
        model : class
            Model class.
        tkwargs : dict
            Dictionary of keyword arguments for the model.
        opt : str
            type of optimization either 'random', 'single' or 'multi'


        """ 
        available_models = ['Sobol','GPEI','GPKG','GPMES','Factorial','FullyBayesian','FullyBayesianMOO','FullyBayesian_MTGP','FullyBayesianMOO_MTGP','Thompson','BO','BoTorch','EB','Uniform','MOO','MOO_Modular','ST_MTGP','ALEBO','BO_MIXED','ST_MTGP_NEHVI','ALEBO_Initializer','Contextual_SACBO']
        # not implemented yet
        not_implemented = ['ST_MTGP_NEHVI','ALEBO_Initializer','Contextual_SACBO','ALEBO','GPMES','Thompson','EB','Factorial']

        if estimator in not_implemented:
            raise ValueError('The estimator {} is not implemented yet'.format(estimator))

        # Check if CUDA is available
        if use_CUDA == False:
            tkwargs  = {}
        else:
            tkwargs = {"torch_device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),"torch_dtype": torch.double}

        opt = None
        # Note that we compare the lower case of the estimator to make it more flexible for the user
        if estimator.lower() == 'Sobol'.lower(): # better for initial exploration not for optimization
            model = Models.SOBOL
            tkwargs = {'seed':None}
            opt = 'random'
        elif estimator.lower() == 'Uniform'.lower(): # better for initial exploration not for optimization
            model = Models.UNIFORM
            tkwargs = {} # no need for tkwargs for this model
            opt = 'random'
        # elif estimator.lower() == 'Factorial'.lower(): # better for initial exploration not for optimization
        #     model = Models.FACTORIAL
        #     tkwargs = {} # no need for tkwargs for this model
        #     opt = 'random'

        ### Single objective models
        elif estimator.lower() == 'GPEI'.lower():
            model = Models.GPEI
            opt = 'single'
        elif estimator.lower() == 'GPKG'.lower():
            model = Models.GPKG
            opt = 'single'
        elif estimator.lower() == 'FullyBayesian'.lower():
            model = Models.FULLYBAYESIAN
            opt = 'single'
        elif estimator.lower() == 'BO'.lower():
            model = Models.BOTORCH
            opt = 'single'
        elif estimator.lower() == 'BoTorch'.lower():
            model = Models.BOTORCH_MODULAR
            opt = 'single'
        elif estimator.lower() == 'BO_MIXED'.lower():
            model = Models.BO_MIXED
            opt = 'single'
        
        ### Multi-objective models
        elif estimator.lower() == 'FullyBayesianMOO'.lower():
            model = Models.FULLYBAYESIANMOO
            opt = 'multi'
        elif estimator.lower() == 'FullyBayesianMOO_MTGP'.lower():
            model = Models.FULLYBAYESIANMOO_MTGP
            opt = 'multi'
        elif estimator.lower() == 'ST_MTGP_NEHVI'.lower():
            model = Models.ST_MTGP_NEHVI
            opt = 'multi'
        elif estimator.lower() == 'MOO'.lower():
            model = Models.MOO
            opt = 'multi'
        elif estimator.lower() == 'MOO_Modular'.lower():
            model = Models.MOO_MODULAR
            opt = 'multi'
       
        ### Not yet implemented
        # elif estimator.lower() == 'GPMES'.lower():
        #     model = Models.GPMES
        # elif estimator.lower() == 'Thompson'.lower():
        #     model = Models.THOMPSON
        #     tkwargs = {}
        #  elif estimator.lower() == 'FullyBayesian_MTGP'.lower():
        #     model = Models.FULLYBAYESIAN_MTGP
        # elif estimator.lower() == 'Factorial'.lower(): ## probably just need a ChoiceParameter or FixedParameter in the search space need to test later
        #     model = Models.FACTORIAL
        #     tkwargs = {} # no need for tkwargs for this model
        # elif estimator.lower() == 'EB'.lower():
        #     model = Models.EMPERICAL_BAYES_THOMPSON
        # elif estimator.lower() == 'ST_MTGP'.lower():
        #     model = Models.ST_MTGP
        # elif estimator.lower() == 'ALEBO'.lower():
        #     model = Models.ALEBO
        # elif estimator.lower() == 'ALEBO_Initializer'.lower():
        #     model = Models.ALEBO_Initializer
        # elif estimator.lower() == 'Contextual_SACBO'.lower():
        #     model = Models.Contextual_SACBO
        else:
            raise ValueError('estimator must be one of the following: {}'.format(available_models))
        
        return model, tkwargs, opt

    def makeobjectives(self, targets, obj_type='MSE', threshold = 1000,is_MOO=False):
        """Convert the targets to the format required by the Ax/Botorch library

        Parameters
        ----------
        targets : list of dict
            list of dictionaries with the following keys:\\
                'model': a pointer to a function y = f(X) where X has m dimensions\\
                'data': dictionary with keys\\
                'X':ndarray with shape (n,m) where n is the number of evaluations for X \\
                'y':ndarray with shape (n,)\\
                'X_dimensions': list of string: the names of the dimensions in X\\
                'X_units': list of string: the units of the dimensions in X\\
                'y_dimension': string: the name of the dimension y\\
                'y_unit': string: the unit of the dimension y\\
                'weight': float: the weight of the target\\
                'loss': string: the loss function to be used\\
                'threshold': float: the threshold for the loss function\\
        obj_type : str, optional
            the type of objective function to be used, by default 'MSE'
        loss : str, optional
            the loss function to be used, by default 'linear'
        threshold : float, optional
            the threshold for the loss function, by default 1000

        Returns
        -------
        list of Metric() objects
            list of Metric() objects

        """
        # if threshold is not a list, make it a list
        if threshold is None: # if threshold is None, then set it to None for all targets and Ax will infer the threshold
            threshold =  [None]*len(targets)
        elif not isinstance(threshold,list):
            threshold = threshold*np.ones(len(targets))

        for idx,t in enumerate(targets):
            if 'threshold' in t.keys():
                threshold[idx] = t['threshold']
        
        # save the threshold in the targets
        for idx,t in enumerate(targets):
            t['threshold'] = threshold[idx]


        # check if the length of the threshold list is the same as the number of metrics
        if len(threshold) != len(targets):
            raise ValueError('Threshold must either be a float or a list with the same length as the number of targets')

        AxObjectives,metrics_names = {},[]
        
        if is_MOO:
            for _, target in enumerate(targets):
                if 'minimize' in target.keys():
                    minimize = target['minimize']
                    if not isinstance(minimize, bool): # check if minimize is a boolean
                        warnings.warn('minimize for target '+str(_)+' must be a boolean. Using default value of True.')
                        minimize = True
                else:
                    minimize = True
                if 'obj_type' in target.keys():
                    AxObjectives[target['obj_type']+'_'+target['target_name']] = ObjectiveProperties(minimize=minimize, threshold= threshold[_])
                else:
                    AxObjectives[obj_type+'_'+target['target_name']] = ObjectiveProperties(minimize=minimize, threshold= threshold[_]) 

                metrics_names.append(obj_type+'_'+target['target_name'])
        else:
            mini = []
            for _, target in enumerate(targets):
                if 'minimize' in target.keys():
                    mini.append(target['minimize'])
            # check if all targets have the same minimize value
            if mini == []:
                minimize = True
            else:
                if len(set(mini)) == 1:
                    minimize = mini[0]
                else:
                    raise ValueError('All targets must have the same minimize value')

            AxObjectives[obj_type] = ObjectiveProperties(minimize=minimize, threshold= threshold[0])
            metrics_names.append(obj_type)

        return AxObjectives,metrics_names
    
    def evaluate(self,px,obj_type,loss,threshold=1,is_MOO=False):
        """Evaluate the target at a given set of parameters

        Parameters
        ----------
        px : list
            list of parameters values
        obj_type : str
            type of objective function to be used, see self.obj_func_metric
        loss : str
            loss function to be used, see self.lossfunc
        threshold : float, optional
            threshold for the loss function, by default 1
        is_MOO : bool, optional
            whether to use multi-objective optimization or enforce single-objective optimization, by default False

        Returns
        -------
        float
            the model output
        """    

        pnames = [p.name for p in self.params if p.relRange != 0]

        px_ = [px[pnames[i]] for i in range(len(pnames))]

        self.params_w(px_,self.params)

        zs = [] # list of losses
        target_weights = []
        res = {}
        if is_MOO:
            for num, t in enumerate(self.targets):
                X = t['data']['X']
                y = t['data']['y']
                if 'weight' in t.keys(): 
                    weight = t['weight']
                else:
                    weight = 1
                yf = t['model'](X,self.params)

                if 'loss' in t.keys(): # if loss is specified in targets, use it
                    loss = t['loss']

                if 'obj_type' in t.keys():
                    z = self.obj_func_metric(t,yf,obj_type=t['obj_type'])
                    res[t['obj_type']+'_'+t['target_name']] = self.lossfunc(z,loss,threshold=1)
                else:
                    z = self.obj_func_metric(t,yf,obj_type=obj_type)
                    res[obj_type+'_'+t['target_name']] = self.lossfunc(z,loss,threshold=1) # threshold is set to 1 as we don't need it for MOO and if it is a list it will not work

            return res
        else:
            for num, t in enumerate(self.targets):
                X = t['data']['X']
                y = t['data']['y']
                if 'weight' in t.keys(): 
                    weight = t['weight']
                else:
                    weight = 1
                yf = t['model'](X,self.params)

                z = self.obj_func_metric(t,yf,obj_type=obj_type)

                if 'loss' in t.keys(): # if loss is specified in targets, use it
                    loss = t['loss']

                if 'threshold' in t.keys(): # if threshold is specified in targets, use it
                    threshold = t['threshold']
                
                if 'target_weight' in t.keys() and (isinstance(t['target_weight'], float) or isinstance(t['target_weight'], int)): # if target_weight is specified in targets, use it
                    #check if target_weight is a float
                    # if isinstance(t['target_weight'], float) or isinstance(t['target_weight'], int):
                        target_weights.append(t['target_weight'])
                else:
                    target_weights.append(1)
                    warnings.warn('target_weight for target '+str(num)+' must be a float or int. Using default value of 1.')

                zs.append(self.lossfunc(z,loss,threshold=threshold))

            # cost is the weigthed average of the losses
            if len(zs) == 1:
                cost = zs[0]
            else:
                cost = np.average(zs, weights=target_weights)    

            return {obj_type:cost}
    
    @abstractmethod
    def evaluate_custom(self,px,obj_type,loss,threshold=1,is_MOO=False):
        """ Create a custom evaluation function that can be used with the Ax/Botorch library and needs to be implemented by the user
        should return a dictionary with the following format:
        {'metric_name':metric_value}

        Parameters
        ----------
        px : list
            list of parameters values
        obj_type : str
            type of objective function to be used, see self.obj_func_metric
        loss : str
            loss function to be used, see self.lossfunc
        threshold : float, optional
            threshold for the loss function, by default 1
        is_MOO : bool, optional
            whether to use multi-objective optimization or enforce single-objective optimization, by default False
        """     
        raise NotImplementedError('Please implement your own evaluate_custom function')   

    
    def BoTorchOpti(self,n_jobs=[4,4], n_step_points = [5, 10], models=['Sobol','GPEI'], obj_type='MSE',loss='linear',threshold=100, model_kwargs_list = None, model_gen_kwargs_list = None,use_CUDA=True,is_MOO=False,use_custom_func=False,suggest_only=False,show_posterior=True,kwargs_posterior=None,verbose=True):
        """Optimize the model using the Ax/Botorch library
        Uses the Expected Hypervolume Improvement (EHVI) algorithm

        Parameters
        ----------
        n_jobs : list, optional
            number of parallel jobs for each step, by default [4,4]
        n_step_points : list, optional
            number of points to sample for each step, by default [5, 10]
        models : list, optional
            list of models to use for each step, by default ['Sobol','GPEI']
        model_kwargs_list : list, optional
            list of dictionaries of model kwargs to use for each step, by default None
            Can contains :
            'surrogate' : Surrogate model to use. 
            'botorch_acqf_class' : BoTorch acquisition function class to use.
        model_gen_kwargs_list : list, optional
            list of dictionaries of model generation kwargs to use for each step, by default None
        use_CUDA : bool, optional
            whether to use CUDA or not, by default True
        is_MOO : bool, optional
            whether to use multi-objective optimization or enforce single-objective optimization, by default False
        use_custom_func : bool, optional
            use a custom evaluation function instead of the default one, this is useful when the same model is used for different targets, by default False
        suggest_only : bool, optional
            only suggest the next point and does not evaluate it, by default False
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
            whether to print the optimization steps or not, by default True

        Returns
        -------
        AxClient
            the AxClient object

        Raises
        ------
        ValueError
            if n_jobs, n_step_points and models are not lists of the same length

        """    

        self.params_r(self.params) 
        
        num_trials = sum(n_step_points) # total number of trials

        # check that n_jobs, n_step_points and models are lists of the same length
        if not (len(n_jobs) == len(n_step_points) == len(models)):
            raise ValueError('n_jobs, n_step_points and models must be lists of the same length')
        if model_kwargs_list is not None:
            if not len(model_kwargs_list) == len(models):
                raise ValueError('model_kwargs_list must be a list of the same length as models, please provide a dictionary of kwargs for each model even if empty')

        # check that no n_jobs is larger than os.cpu_count()-1, reset to os.cpu_count()-1 if so and raise warning
        for i in range(len(n_jobs)):
            if n_jobs[i] > os.cpu_count()-1 and self.parallel: 
                n_jobs[i] = os.cpu_count()-1
                warnings.warn('n_jobs for step '+str(i)+' is larger than os.cpu_count()-1. Resetting to '+str(os.cpu_count()-1))
        maxjobs = max(n_jobs) # maximum number of parallel jobs

        # make list of models and tkwargs for each step
        steps, optitype = [], []
        for i in range(len(n_jobs)):
            dum_model, dum_tkwargs, opt = self.get_model(estimator=models[i],use_CUDA=use_CUDA)
            optitype.append(opt)
            if model_kwargs_list is not None: # if model_kwargs_list is specified, update dum_tkwargs
                for key in model_kwargs_list[i].keys():
                    dum_tkwargs[key] = model_kwargs_list[i][key]
            if model_gen_kwargs_list is not None: # if model_gen_kwargs_list is specified, update dum_tkwargs
                dum_tkwargs_gen = model_gen_kwargs_list[i]
            else:
                dum_tkwargs_gen = {}
            steps.append(GenerationStep(model=dum_model,num_trials=n_step_points[i],min_trials_observed=n_step_points[i],max_parallelism=n_jobs[i],model_kwargs = dum_tkwargs, model_gen_kwargs = dum_tkwargs_gen))

        if 'single' in optitype and 'multi' in optitype: # if both single and multi-objective optimization are specified, raise error
            raise ValueError('Cannot mix single and multi-objective optimization')
        # Note that is you have more than one target but is_MOO is False, the targets will be averaged and the loss will be computed on the weighted average

        generation_strategy = GenerationStrategy(steps=steps,) # initialize the generation strategy

        ax_client = AxClient(generation_strategy=generation_strategy,enforce_sequential_optimization=False,verbose_logging=verbose) # initialize the AxClient

        parameters = self.ConvertParams(self.params)

        objectives,metrics_name = self.makeobjectives(self.targets, obj_type=obj_type, threshold = threshold,is_MOO=is_MOO) # make the objectives
        
        # if self.parameter_constraints is not None:


        ax_client.create_experiment(
            name = 'BOAR_torch',
            parameters=parameters,
            # objective_name='MSE',
            # minimize=True,
            objectives=objectives,
            # Sets max parallelism to 10 for all steps of the generation strategy.
            choose_generation_strategy_kwargs={
                "num_trials": num_trials,
                "max_parallelism_override": maxjobs,
                "enforce_sequential_optimization": False,},
            parameter_constraints = self.parameter_constraints,
        )


        # run the optimization
        n = 0
        hv_list = []
        cum_n_step_points = np.cumsum(n_step_points)
        # check if any n_jobs is larger than 1, if so use joblib to parallelize the evaluation of the custom function

        
        # if use_custom_func == True and any(np.asarray(n_jobs) > 1):
        #     n_jobs = [1 for i in range(len(n_jobs))]
        #     warnings.warn('n_jobs must be 1 when using a custom evaluation function, setting n_jobs to 1')
            # use ray to parallelize the evaluation of the custom function, for some reason it does not work with joblib
            # ray.init(num_cpus=maxjobs)
        
        if 'recall' in self.warmstart:
            try:
                # load the old data
                path2file = self.Path2OldXY
                with open(path2file, "r") as outfile:
                    old_xy_dict = json.load(outfile)

                self.old_xy['x'] = old_xy_dict['x']
                for num, t in enumerate(self.targets): # save the old y values for each target
                    self.old_xy['y_'+str(num)] = old_xy_dict['y_'+str(num)]
                    self.old_xy['ydyn_'+str(num)] = old_xy_dict['ydyn_'+str(num)]
                
                # add the old data to the Ax client
                pnames = [p.name for p in self.params if p.relRange != 0]
                valtypes = [p.val_type for p in self.params if p.relRange != 0]
                metrics_names = [obj_type+'_'+t['target_name'] if 'obj_type' not in t.keys() else t['obj_type']+'_'+t['target_name'] for t in self.targets]
                target_weights = []
                for i in range(len(self.old_xy['x'])):
                    dic,dic_metric = {},{}
                    for j in range(len(pnames)):
                        if valtypes[j] == 'str':
                            dic[pnames[j]] = str(self.old_xy['x'][i][j])
                        elif valtypes[j] == 'int':
                            dic[pnames[j]] = int(self.old_xy['x'][i][j])
                        else:
                            dic[pnames[j]] = float(self.old_xy['x'][i][j])

                    if is_MOO:
                        for num, t in enumerate(self.targets):
                            dic_metric[metrics_names[num]] = (self.old_xy['y_'+str(num)][i],None)
                        parameters, trial_index = ax_client.attach_trial(parameters=dic)
                        ax_client.complete_trial(trial_index=trial_index, raw_data=dic_metric)
                    else:
                        zs = [] # list of losses
                        for num, t in enumerate(self.targets):
                            zs.append(self.old_xy['y_'+str(num)][i])
                            if 'target_weight' in t.keys() and (isinstance(t['target_weight'], float) or isinstance(t['target_weight'], int)): # if target_weight is specified in targets, use it
                                #check if target_weight is a float
                                # if isinstance(t['target_weight'], float) or isinstance(t['target_weight'], int):
                                    target_weights.append(t['target_weight'])
                            else:
                                target_weights.append(1)
                                warnings.warn('target_weight for target '+str(num)+' must be a float or int. Using default value of 1.')

                        # cost is the weigthed average of the losses
                        if len(zs) == 1:
                            cost = zs[0]
                        else:
                            cost = np.average(zs, weights=target_weights)
                        parameters, trial_index = ax_client.attach_trial(parameters=dic)
                        ax_client.complete_trial(trial_index=trial_index, raw_data=(cost,None))

            except Exception as e:
                print('Could not load the old_xy data so we keep going without it')
                print('error message: '+str(e))



        while n < num_trials:
            curr_batch_size = int(n_jobs[np.argmax(cum_n_step_points>n)]) # number of trials in the current batch
            n = n + curr_batch_size
            if n > num_trials: # if the number of trials is larger than the total number of trials, reduce the batch size
                curr_batch_size = int(curr_batch_size - (n - num_trials))
                n = num_trials

            trial_mapping, optimization_complete = ax_client.get_next_trials(curr_batch_size)

            
            if not suggest_only:
                if use_custom_func == False:
                    if curr_batch_size == 1 or not self.parallel: # do in serial
                        results = [self.evaluate(px,obj_type,loss,threshold=threshold,is_MOO=is_MOO) for px in trial_mapping.values()]
                    else: # do in parallel with joblib
                        results = Parallel(n_jobs=curr_batch_size)(delayed(self.evaluate)(px,obj_type,loss,threshold=threshold,is_MOO=is_MOO) for px in trial_mapping.values())
                else:
                    warnings.warn('Using custom evaluation function, make sure that the function returns a dictionary with the following format: {metric_name:metric_value}')
                    if curr_batch_size == 1 or not self.parallel:
                        results = [self.evaluate_custom(px,obj_type,loss,threshold=threshold,is_MOO=is_MOO) for px in trial_mapping.values()]
                    else:
                    #     raise ValueError('Cannot use custom evaluation function with more than one job, please set n_jobs to 1')
                        results = Parallel(n_jobs=curr_batch_size)(delayed(self.evaluate_custom)(px,obj_type,loss,threshold=threshold,is_MOO=is_MOO) for px in trial_mapping.values())
                    # # user ray 
                    # results = ray.get([self.evaluate_custom.remote(px,obj_type,loss,threshold=threshold,is_MOO=is_MOO) for px in trial_mapping.values()])


                # report the completion of trials to the Ax client
                for trial_index, raw_data in zip(trial_mapping.keys(), results):
                    ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
            else:
                # mark as failed
                for trial_index in trial_mapping.keys():
                    ax_client.abandon_trial(trial_index=trial_index)

            # get the hypervolume
            if is_MOO:
                try:
                    hv = ax_client.get_hypervolume()
                    hv_list.append(hv)
                except Exception as e:
                    hv = 0
                    hv_list.append(hv)
                    # warnings.warn('Could not compute hypervolume, error message: '+str(e))

        # Collect the old xy data for future use
        if 'collect' in self.warmstart:
            dic = {}
            if len(self.old_xy['x']) > 0 and not 'collect_BO' in self.warmstart: # if there is old data, save it3
                dic['x'] = self.old_xy['x']
                for num, t in enumerate(self.targets): # save the old y values for each target
                    dic['y_'+str(num)] = self.old_xy['y_'+str(num)]
                    dic['ydyn_'+str(num)] = self.old_xy['ydyn_'+str(num)]
            else: # if there is no old data, save the data from the BO
                dic['x'] = []
                for num, t in enumerate(self.targets): # save the old y values for each target
                    dic['y_'+str(num)] = []
                    dic['ydyn_'+str(num)] = []
            triedX = ax_client.generation_strategy.trials_as_df
            triedY = ax_client.experiment.fetch_data().df
            metric_names = triedY['metric_name'].unique() # get the metric names

            for num, t in enumerate(self.targets): # save the old y values for each target
                xdum, ydum, ydyndum = [], [], []
                ydyndum = 1
                triedYdum = triedY[triedY['metric_name']==metric_names[num]]
                for trial_index in triedX['Trial Index']:
                    dum = triedX[triedX['Trial Index']==trial_index]['Arm Parameterizations'].values[0]
                    key = list(dum)[0]
                    xdum.append(list(dum[key].values()))
                    ydum.append(triedYdum[triedYdum['trial_index']==trial_index]['mean'].values[0])
                dic['y_'+str(num)].extend(ydum)
                dic['ydyn_'+str(num)] = ydyndum
            dic['x'].extend(xdum)
            path2file = self.SaveOldXY2file # path to file
            with open(path2file, "w") as outfile:
                json.dump(dic, outfile)
                

        # get the best parameters
        try:
            if not is_MOO:
                best_parameters, values = ax_client.get_best_parameters()
            else:
                self.hv_list = hv_list
                try:
                    pareto = ax_client.get_pareto_optimal_parameters(use_model_predictions=False)
                    keys = list(pareto.keys())
                    best_parameters = pareto[keys[-1]][0]

                    # Make a list of the pareto points in the for [key, MSE_trMC, MSE_trPL] for easier handling
                    pareto_list=[]
                    for i, sublist in zip(pareto.keys(), pareto.values()):
                        dum_list = [i]
                        for j in range(len(sublist[1][0].keys())):
                            dum_list.append(sublist[1][0][list(sublist[1][0].keys())[j]])
                        pareto_list.append(dum_list)

                    # Sort the list based on the first and second elements of each sublist
                    # Calculate squared sum of indices for each element and find the one with the lowest sum
                    min_index_sum = float('inf')
                    min_index_element = None
                    indexs = []
                    for i, sublist in enumerate(pareto_list):
                        dum_list = []
                        for j in range(len(sublist)):
                            if j > 0:
                                dum_list.append(sorted(pareto_list, key=lambda x: x[j]).index(sublist))
                        indexs.append(dum_list)
                    # Find the index of the element with the lowest sum of indices
                    sum_indexs = [sum(i) for i in zip(*indexs)]
                    min_index_element = sum_indexs.index(min(sum_indexs))
                    best_parameters = pareto[pareto_list[min_index_element][0]][0]
                    
                except Exception as e:
                    raise ValueError('No pareto optimal parameters found, try to increase the number of trials or the threshold or let Ax infer the threshold by setting it to None.\n',e)

            px = [best_parameters[p.name] for p in self.params if p.relRange != 0]
            self.params_w(px,self.params)

            if not is_MOO:
                # number of data points
                Num_data_pts = 0
                for num,t in enumerate(self.targets): # get the number of data points
                    Num_data_pts += len(t['data']['y'])


                triedX = ax_client.generation_strategy.trials_as_df

                # get previous trials
                points = []
                for index, row in triedX.iterrows():
                    dum = row['Arm Parameterizations']
                    key = list(dum.keys())[0]
                    points.append(list(dum[key].values()))

                ax_client.fit_model() # fit the model
                gpr = ax_client.generation_strategy.model # get the model
                xmin, funmin = self.expected_minimum_BOAR(triedX,gpr)

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

                # save parameters to self for later use
                self.N = Num_data_pts
                self.gpr = gpr
                self.fscale = None
                self.beta_scaled = beta
                self.points = points
                self.kwargs_posterior = kwargs_posterior

                # get the posterior probabiliy distribution p(w|t)
                if show_posterior:
                    
                    pf = [pp for pp in self.params if pp.relRange!=0]
                    p0, lb_main, ub_main = self.params_r(self.params)

                    xmin0,std = self.posterior(pf, lb_main, ub_main,points=points,beta_scaled = beta,N=Num_data_pts,gpr = gpr,fscale=None,kwargs_posterior=kwargs_posterior)

                    # Note: the posterior is calculated with from the surrogate function and not the ground truth function therefore it is not always accurate
                    #       especially when the surrogate function is not trained well. This is why the best fit parameters are taken from the best one sampled by the BO
                    #       and not from the posterior. The posterior is only used to get the error bars. This mean that sometimes the best fit parameters is not necessarily
                    #       the one with the highest posterior probability. In this case the 95% confidence interval that is outputted is stretched to include the best fit parameters.
                    #       This is not a problem because the posterior is not used to get the best fit parameters but only to get the error bars, this way guarantee that the best fit is 
                    #       always within the outputted error bars.
                    #       The 95% interval is stored in std (so std is NOT the standard deviation of the posterior distribution)
                
                    self.params_w(px,self.params,std=std) # read out Fitparams & respect settings
        except Exception as e:
            if suggest_only:
                warnings.warn('Could not get the best parameters as suggest_only is set to True and no trial was completed, error message: '+str(e))
            else:
                raise ValueError('Error message: '+str(e))
        return ax_client

    
    ###############################################################################
    ############################## Plot utils #####################################
    ###############################################################################
    def plot_all_objectives(self, ax_client,**kwargs):
        """Plot all objectives 

        Parameters
        ----------
        ax_client : AxClient() object
            AxClient() object
        kwargs : dict, optional
            keyword arguments for the plot, by default {}
                savefig : boolean, optional
                    save the figure, by default False
                savefig_name : str, optional
                    name of the file to save the figure, by default ''objectives'
                savefig_dir : str, optional
                    directory to save the figure, by default self.res_dir
                figext : str, optional
                    extension of the figure, by default '.png'
                figsize : tuple, optional
                    size of the figure, by default (5*nb_params,5*nb_params)
                figdpi : int, optional
                    dpi of the figure, by default 300

        Returns
        -------
        None

        """ 

        figsize = kwargs.get('figsize',(15,15))
        figdpi = kwargs.get('figdpi',300)
        figext = kwargs.get('figext','.png')
        savefig = kwargs.get('savefig',False)
        savefig_name = kwargs.get('savefig_name','objectives')
        logscale = kwargs.get('logscale',False)

        cm = plt.cm.get_cmap('viridis')
        df = exp_to_df(ax_client.experiment).sort_values(by=['trial_index'])
        metric_names = ax_client.experiment.fetch_data().df['metric_name'].unique()

        if len(metric_names) == 1:
            fig, axes =  plt.subplots(1,1,figsize=figsize)
            train_obj  = df[metric_names[0]].values
            batch_number = df.trial_index.values
            sc = axes.plot(batch_number, train_obj)
            axes.set_xlabel('Iteration')
            axes.set_ylabel(metric_names[0])
            if logscale:
                axes.set_yscale('log')

        elif len(metric_names) == 2:
            fig, axes =  plt.subplots(1,1,figsize=figsize)

            train_obj  = df[[metric_names[0],metric_names[1]]].values
            batch_number = df.trial_index.values
            sc = axes.scatter(train_obj[:, 0], train_obj[:,1], c=batch_number, alpha=0.8)
            axes.set_xlabel(metric_names[0])
            axes.set_ylabel(metric_names[1])
            norm = plt.Normalize(batch_number.min(), batch_number.max())
            sm =  ScalarMappable(norm=norm, cmap=cm)
            sm.set_array([])
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.9, 0.15, 0.01, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.ax.set_title("Iteration")
            if logscale:
                axes.set_xscale('log')
                axes.set_yscale('log')

        else:
            fig, axes =  plt.subplots(len(metric_names)-1,len(metric_names)-1,figsize=figsize)
            comb = list(itertools.combinations(metric_names, 2))
            count = 0
            for i in range(len(metric_names)-1):
                for j in range(len(metric_names)-1):

                    if i >= j:
                        metricx = comb[count][0]
                        metricy = comb[count][1]
                        train_obj  = df[[metricx,metricy]].values
                        batch_number = df.trial_index.values
                        sc = axes[i,j].scatter(train_obj[:, 0], train_obj[:,1], c=batch_number, alpha=0.8)

                        axes[i,j].set_xlabel(metricx)
                        axes[i,j].set_ylabel(metricy)
                        if logscale:
                            axes[i,j].set_xscale('log')
                            axes[i,j].set_yscale('log')
                        count += 1
                    elif i < j:
                        axes[i,j].axis('off')
            norm = plt.Normalize(batch_number.min(), batch_number.max())
            sm =  ScalarMappable(norm=norm, cmap=cm)
            sm.set_array([])
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.9, 0.15, 0.01, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.ax.set_title("Iteration")

        plt.show()

        if savefig:
            fig.savefig(os.path.join(self.res_dir,savefig_name+figext),dpi=figdpi)




    def plot_density(self, ax_client,**kwargs):
        """ Plot the density of the of the points in the search space

        Parameters
        ----------
        ax_client : AxClient() object
            AxClient() object
        kwargs : dict, optional
            keyword arguments for the plot, by default {}
                savefig : boolean, optional
                    save the figure, by default False
                savefig_name : str, optional
                    name of the file to save the figure, by default ''objectives'
                savefig_dir : str, optional
                    directory to save the figure, by default self.res_dir
                figext : str, optional
                    extension of the figure, by default '.png'
                figsize : tuple, optional
                    size of the figure, by default (5*nb_params,5*nb_params)
                figdpi : int, optional
                    dpi of the figure, by default 300

        Raises
        ------
        ValueError
            axis_type must be either log or linear

        """      

        figsize = kwargs.get('figsize',(15,15))
        figdpi = kwargs.get('figdpi',300)
        figext = kwargs.get('figext','.png')
        savefig = kwargs.get('savefig',False)
        savefig_name = kwargs.get('savefig_name','points_density')

        experiment = ax_client.experiment

        pnames = [p.name for p in self.params if p.relRange != 0 and p.val_type != 'str'] # get the parameter names don't include the string parameters
        pnames_display = [p.display_name for p in self.params if p.relRange != 0 and p.val_type != 'str']
        lims_low = [p.lims[0] for p in self.params if p.relRange != 0]
        lims_high = [p.lims[1] for p in self.params if p.relRange != 0]
        p_axis_type = [p.axis_type for p in self.params if p.relRange != 0]

        df = exp_to_df(experiment).sort_values(by=["trial_index"])

        df2 = df[pnames]

        # # udpate values in df2 if optim_type is log
        for i in range(len(self.params)):
        # for i in range(len(pnames)):
            if self.params[i].relRange != 0:
                if self.params[i].optim_type == 'log':
                    df2[self.params[i].name] = 10**(df2[self.params[i].name]) # rescale to original value
                elif self.params[i].optim_type == 'linear':
                    df2[self.params[i].name] = df2[self.params[i].name]*self.params[i].p0m # rescale to original value
                else:
                    raise ValueError('axis_type must be either log or linear')

        if len(pnames) == 1:
            fig, axes = plt.subplots(1,1,figsize=figsize)
            if p_axis_type[0] == 'log':
                axes.set_xscale('log')
            axes.set_xlim(lims_low[0],lims_high[0])

            sns.histplot(df2[pnames[0]],ax=axes,bins=20,stat='probability',kde=True)

            axes.tick_params(axis='x', rotation=45)
            axes.tick_params(axis='x', which='minor', rotation=45)
            axes.set_xlabel(pnames_display[0])
            axes.set_ylabel('Probability')
        else:
            # make a figure with subplots for each parameter combination
            fig, axes = plt.subplots(len(pnames), len(pnames), figsize=figsize)
            for i in range(len(pnames)):
                for j in range(len(pnames)):
                    if i == j:
                        if p_axis_type[i] == 'log':
                            axes[i,j].set_xscale('log')
                        axes[i,j].set_xlim(lims_low[i],lims_high[i])

                        sns.histplot(df2[pnames[i]],ax=axes[i,j],bins=20,stat='probability',kde=True)

                        if i != len(pnames)-1:
                            axes[i,j].tick_params(axis='x', rotation=45)
                            axes[i,j].tick_params(axis='x', which='minor', rotation=45)
                            axes[i,j].set_xlabel(pnames_display[i])
                            axes[i,j].xaxis.set_label_position('top')
                            axes[i,j].xaxis.tick_top()
                            axes[i,j].yaxis.set_label_position('right')
                            axes[i,j].set_yticklabels([])
                            # axes[i,j].yaxis.tick_right()
                            # axes[i,j].set_ylabel('Probability')
                            axes[i,j].set_ylabel('')

                        else:
                            axes[i,j].tick_params(axis='x', rotation=45)
                            axes[i,j].tick_params(axis='x', which='minor', rotation=45)
                            axes[i,j].set_xlabel(pnames_display[i])
                            axes[i,j].yaxis.set_label_position('right')
                            # remove the yticks
                            axes[i,j].set_yticklabels([])
                            # axes[i,j].yaxis.tick_right()
                            # axes[i,j].set_ylabel('Probability')
                            axes[i,j].set_ylabel('')


                    elif i > j:
                        if p_axis_type[j] == 'log':
                            axes[i,j].set_xscale('log')
                        if p_axis_type[i] == 'log':
                            axes[i,j].set_yscale('log')
                        
                        axes[i,j].set_xlim(lims_low[j],lims_high[j])
                        axes[i,j].set_ylim(lims_low[i],lims_high[i])


                        sns.kdeplot(x=df2[pnames[j]],y=df2[pnames[i]],ax=axes[i,j],shade=True,cmap='viridis',levels=20)
                        axes[i,j].scatter(df2[pnames[j]],df2[pnames[i]],s=5,c='k',alpha=0.5)

                        if i == len(pnames)-1:
                            axes[i,j].set_xlabel(pnames_display[j])
                            axes[i,j].tick_params(axis='x', rotation=45)
                            axes[i,j].tick_params(axis='x', which='minor', rotation=45)
                        else:
                            axes[i,j].set_xticklabels([])
                            axes[i,j].set_xticklabels([], minor=True)
                            axes[i,j].set_xlabel('')

                        if j == 0:
                            axes[i,j].set_ylabel(pnames_display[i])
                        else:
                            axes[i,j].set_yticklabels([])
                            axes[i,j].set_yticklabels([], minor=True)
                            axes[i,j].set_ylabel('')
                            
                        
                        
                    else:
                        axes[i,j].axis('off')

        plt.tight_layout()

        if savefig:
            fig.savefig(os.path.join(self.res_dir,savefig_name+figext),dpi=figdpi)

    def plot_hypervolume(self,hv_list=None,**kwargs):
        """Plot the hypervolume trace

        Parameters
        ----------
        hv_list : list, optional
            list of hypervolumes, by default None
        kwargs : dict, optional
            keyword arguments for the plot, by default {}
                savefig : boolean, optional
                    save the figure, by default False
                savefig_name : str, optional
                    name of the file to save the figure, by default ''objectives'
                savefig_dir : str, optional
                    directory to save the figure, by default self.res_dir
                figext : str, optional
                    extension of the figure, by default '.png'
                figsize : tuple, optional
                    size of the figure, by default (5*nb_params,5*nb_params)
                figdpi : int, optional
                    dpi of the figure, by default 300
                logscale : bool, optional
                    use logscale, by default False
        """        

        figsize = kwargs.get('figsize',(15,15))
        figdpi = kwargs.get('figdpi',300)
        figext = kwargs.get('figext','.png')
        savefig = kwargs.get('savefig',False)
        savefig_name = kwargs.get('savefig_name','hypervolume')
        logscale = kwargs.get('logscale',False)


        if hv_list is None:
            hv_list = self.hv_list

        hv_list = np.asarray(hv_list)
        maxHV = 1
        for t in self.targets:
            maxHV = maxHV*t['threshold']
            
        figsize = kwargs.get('figsize',(15,15))
        figdpi = kwargs.get('figdpi',300)
        figext = kwargs.get('figext','.png')
        savefig = kwargs.get('savefig',False)
        savefig_name = kwargs.get('savefig_name','hypervolume')

        fig, axes = plt.subplots(1,1,figsize=figsize)
        axes.plot(maxHV-hv_list)

        if logscale:
            axes.set_yscale('log')

        axes.set_xlabel('Batch iterations')
        axes.set_ylabel('Hypervolume difference')
        plt.tight_layout()
        plt.show()

    ###############################################################################
    ############################ Posterior utils ##################################
    ###############################################################################
    # Note that this is only for single objective optimization
    # While this part could be put in the optimizer.py main file, it is put here to give freedom to the user not to install the torch and Ax packages and only install the scikit-optimize package. In the future, this could be moved to the optimizer.py file.

    def expected_minimum_BOAR(self, triedX, gpr, n_random_starts=20, random_state=None):
        """Compute the minimum over the predictions of the last surrogate model.
        
        Note: that this will be useful only from single objective optimization and if the goal is to minimize the surrogate function.

        This was adapted from the scikit-optimize package. The original code can be found here:
        [scikit-optimize](https://scikit-optimize.github.io/stable/index.html) in the file scikit-optimize/skopt/utils.py

        Parameters
        ----------
        ax_client : AxClient
            AxClient object.
        n_random_starts : int, default=20
            Number of points to sample randomly before fitting the surrogate
            model. If n_random_starts=0, then the initial point is taken as the
            best point seen so far (usually the last point in the GP model).
        random_state : int, RandomState instance or None, optional (default=None)
            Set random state to something other than None for reproducible
            results.

        Returns
        -------
        x : ndarray, shape (n_features,)
            The point which minimizes the surrogate function.
        fun : float
            The surrogate function value at the minimum.

        """
        
        p0, lb, ub = self.params_r(self.params)

        # get previous trials
        rx,vals = [],[]
        for index, row in triedX.iterrows():
            dum = row['Arm Parameterizations']
            key = list(dum.keys())[0]
            rx.append(dum[key])
            vals.append(list(dum[key].values()))


        if n_random_starts > 0:
            # Sample some points at random
            for i in range(n_random_starts):
                counter = 0
                dum = {}
                for p in self.params:
                    if p.relRange != 0:
                        dum[p.name] = np.random.uniform(lb[counter],ub[counter])
                        counter += 1
                rx.append(dum)
                vals.append(list(dum.values()))
                        
        
        dumdic = rx[0]

        def func(vals): # function to minimize
            for idx,key in enumerate(dumdic.keys()):
                dumdic[key] = vals[idx]

            observation_features = [ObservationFeatures(dumdic)]
            pred = gpr.predict(observation_features)
            pred = pred[0] # get the mean
            key = list(pred.keys()) # get the key
            pred = pred[key[0]][0] # get the value
            return pred

        best_x = None
        best_fun = np.inf

        bounds = tuple(zip(lb,ub)) # bounds for the parameters

        for i in range(len(vals)): # loop over the previous trials
            r = sp_minimize(func,x0=np.asarray(vals[i]),bounds=bounds)
            if r.fun < best_fun:
                best_x = r.x
                best_fun = r.fun
        
        return best_x, best_fun

    def LH_torch_(self,X,beta,N,gpr,fscale=None):
        """Compute the positive log likelihood from the negative log likelihood

        Parameters:
        X : ndarray
            X Data array of size(n,m): n=number of data points, m=number of dimensions
        beta : float
            1 / minimum of scaled surrogate function
        N : integer
            number of data points
        gpr : regressor
            a trained regressor which has a .predict() method
        fscale : float
            scaling factor to keep the surrogate function between 0 and 100 (yields best results in BO but
            here we need the unscaled surrogate function, that is, MSE), Not used here but might be later, default=None
        Returns
        -------
        float
            the positive log likelihood
        """
        rx = []

        for x in X:
            dumdic = {}
            counter = 0
            for p in self.params:
                if p.relRange != 0:
                    dumdic[p.name] = x[counter]
                    counter += 1
            rx.append(dumdic)


        Y = []
        for x in rx: # iterate through all the points to get the prediction
            observation_features = [ObservationFeatures(x)]
            pred = gpr.predict(observation_features)
            pred = pred[0] # get the mean
            key = list(pred.keys()) # get the key
            pred = pred[key[0]][0] # get the value
            Y.append(pred)
        Y = np.asarray(Y)
        SSE = Y*N # the sum of squared errors
        LLH = -beta/2 * SSE + N/2*np.log(beta) - N/2*np.log(2*np.pi) # Bishop eq. 1.62

        return LLH 

    def LH_torch(self,X,beta,N,gpr,fscale=None):
        """Compute the positive log likelihood from the negative log likelihood

        Parameters:
        X : ndarray
            X Data array of size(n,m): n=number of data points, m=number of dimensions
        beta : float
            1 / minimum of scaled surrogate function
        N : integer
            number of data points
        gpr : regressor
            a trained regressor which has a .predict() method
        fscale : float
            scaling factor to keep the surrogate function between 0 and 100 (yields best results in BO but
            here we need the unscaled surrogate function, that is, MSE), Not used here but might be later, default=None
        Returns
        -------
        float
            the positive log likelihood
        """
        rx = []

        for x in X:
            dumdic = {}
            counter = 0
            for p in self.params:
                if p.relRange != 0:
                    dumdic[p.name] = x[counter]
                    counter += 1
            rx.append(dumdic)


        observation_features = [ObservationFeatures(x) for x in rx]
        pred = gpr.predict(observation_features)
        Y = []
        pred = pred[0]
        key = list(pred.keys())
        pred = pred[key[0]]
        for p in pred:
            Y.append(p)
        Y = np.asarray(Y)
        SSE = Y*N # the sum of squared errors
        LLH = -beta/2 * SSE + N/2*np.log(beta) - N/2*np.log(2*np.pi) # Bishop eq. 1.62

        return LLH 

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
                Ninteg = int(1e3)
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
                lh[i,j,:] = self.LH_torch(X,beta_scaled,N,gpr)
        
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
                Ninteg = int(1e3)
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
            lh[i,:] = self.LH_torch(X,beta_scaled,N,gpr)

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
                    aa0prime = self.LH_torch(Xcc[ii],beta_scaled,N,gpr)
                else:
                    aa0prime = np.concatenate((aa0prime, self.LH_torch(Xcc[ii],beta_scaledN,gpr)))
            
            aa1 = aa0prime.reshape(*[Nreso+1 for _ in range(len(pf))]) # bring it into array format
            # Note: the Nreso plus one comes from adding the best fit value
        else:
            aa0 = self.LH_torch(XC,beta_scaled,N,gpr)# predict them all at once
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





# Graveyard


# def EvhiOptimizer(self,n_jobs=1, n_initial_points = 5, n_BO=10, n_jobs_init=None,obj_type='MSE',loss='linear',threshold=100):
#         """Optimize the model using the Ax/Botorch library
#         Uses the Expected Hypervolume Improvement (EHVI) algorithm

#         Parameters
#         ----------
#         n_jobs : int, optional
#             number of parallel jobs, by default 1
#         n_initial_points : int, optional
#             number of initial points, by default 5
#         n_BO : int, optional
#             number of Bayesian Optimization steps, by default 10
#         n_jobs_init : int, optional
#             number of parallel jobs for the initial points, by default None
#         obj_type : str, optional
#             the type of objective function to be used, by default 'MSE'
#         loss : str, optional
#             the loss function to be used, by default 'linear'
#         threshold : float or list, optional
#             the threshold used to calculate the hyper volume, by default 100

#         Returns
#         -------
#         list of dict
#             list of dictionaries with the following keys:\\
#                 'popt': ndarray with shape (m,) where m is the number of variable parameters that are optimized wi


#         """    

#         p0,lb,ub = self.params_r(self.params) # read out Fitparams & respect settings, NEEDED to updateb the self.params for p0m

#         target_names = [target['target_name'] for target in self.targets]
#         if n_jobs_init is None: # if n_jobs_init is not specified, use n_jobs
#             n_jobs_init = n_jobs

#         # if threshold is not a list, make it a list
#         if not isinstance(threshold,list):
#             threshold = threshold*np.ones(len(self.targets))
#         # check if the length of the threshold list is the same as the number of metrics
#         if len(threshold) != len(self.targets):
#             raise ValueError('Threshold must either be a float or a list with the same length as the number of targets')

#         search_space = self.CreateSearchSpace(self.params)

#         objectives,metrics = self.ConvertTargets2Objectives(self.targets,obj_type=obj_type,loss=loss,threshold=threshold)
#         mo = MultiObjective(objectives=objectives,)

        

#         objective_thresholds = [ObjectiveThreshold(metric=metric, bound=val, relative=False)  for metric, val in zip(mo.metrics, threshold)]

#         optimization_config = MultiObjectiveOptimizationConfig(objective = mo,objective_thresholds = objective_thresholds)

#         # create the experiment
#         ehvi_experiment = Experiment(name="MOO_EVHI", search_space=search_space, optimization_config=optimization_config, runner=SyntheticRunner(),)

#         # initialize the experiment
#         ehvi_data = self.initialize_experiment(ehvi_experiment,n_jobs_init, n_initial_points = n_initial_points)
        
#         ehvi_hv_list = []
#         ehvi_model = None
#         N_BATCH = np.ceil(n_BO/n_jobs).astype(int)
#         print(f'Running {N_BATCH} batches of {n_jobs} jobs each')
#         tstart = time()
#         for i in range(N_BATCH):   
#             ehvi_model = get_MOO_NEHVI(
#                 experiment=ehvi_experiment, 
#                 data=ehvi_data,
#             )
#             generator_run = ehvi_model.gen(n_jobs)
#             if n_jobs == 1:
#                 trial = ehvi_experiment.new_trial(generator_run=generator_run)
#             else:
#                 trial = ehvi_experiment.new_batch_trial(generator_run=generator_run)
#             trial.run()
#             totry =[]
#             for i in range(n_jobs):
#                 totry.append(ehvi_experiment.new_batch_trial(generator_run=generator_run))
                
#             # run the trials in parallel with ray
#             # ray.init(num_cpus=n_jobs)
#             # ray.get([trial.run() for trial in totry])
#             # ray.shutdown()

#             # append the outcomes to the list
#             ehvi_data = Data.from_multiple_data([ehvi_data, trial.fetch_data()])
#             # listdata = [ehvi_data]
#             # for trial in totry:
#             #     listdata.append(trial.fetch_data())


#             # ehvi_data = Data.from_multiple_data(listdata)
            
#             exp_df = exp_to_df(ehvi_experiment)

#             outcomes = np.array(exp_df[target_names], dtype=np.double)
#             try:
#                 hv = observed_hypervolume(modelbridge=ehvi_model)
#             except:
#                 hv = 0
#                 print("Failed to compute hv")
#             ehvi_hv_list.append(hv)
#             print(f"Iteration: {i}, HV: {hv:.2e}")
#         print(f'Finished in {time()-tstart} seconds')
#         ehvi_outcomes = np.array(exp_to_df(ehvi_experiment)[target_names], dtype=np.double)
        
#         # get the best parameters
#         data =ehvi_experiment.fetch_data()
#         df = data.df


#         best_arm_name = df.arm_name[df['mean'] == df['mean'].min()].values[0]
#         best_arm = ehvi_experiment.arms_by_name[best_arm_name].parameters
#         best_arm_val = [best_arm[p.name] for p in self.params if p.relRange != 0]
#         xmin = []
#         std = 0*np.ones(len(best_arm_val))
#         idx = 0
#         self.params_w(best_arm_val,self.params,std=std)
#         for par in self.params:
#             if par.relRange != 0:
#                 if par.optim_type == 'log':
#                     xmin.append(10**(best_arm_val[idx]))
#                 else:
#                     xmin.append(best_arm_val[idx]*par.p0m)
#                 idx += 1    
#         self.plot_all_objectives(ehvi_outcomes)
#         self.plot_density(ehvi_experiment)
#         return {'popt':xmin, 'data':df, 'experiment':ehvi_experiment, 'model':ehvi_model, 'outcomes':ehvi_outcomes, 'hv':ehvi_hv_list, 'metric':metrics}
