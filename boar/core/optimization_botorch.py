######################################################################
################ MultiObjectiveOptimizer class #######################
######################################################################
# Authors: 
# Larry Lueer (https://github.com/larryluer)
# Vincent M. Le Corre (https://github.com/VMLC-PV)
# (c) i-MEET, University of Erlangen-Nuremberg, 2021-2022-2023 

# Import libraries
import os,json,copy,warnings
import numpy as np
import pandas as pd
from time import time
from functools import partial
from matplotlib import pyplot as plt
from matplotlib import cm
from copy import deepcopy
from tqdm import tnrange, tqdm_notebook
import seaborn as sns
# sns.set_theme(style="ticks")
# from concurrent.futures import ThreadPoolExecutor
# Import BOtorch and Ax libraries
import torch
from ax import *
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

from ax import Data, Experiment, ParameterType, RangeParameter, SearchSpace
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from ax.core.metric import Metric
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df
from ax.runners.synthetic import SyntheticRunner

# Factory methods for creating multi-objective optimization modesl.
from ax.modelbridge.factory import get_MOO_EHVI, get_MOO_PAREGO, get_MOO_NEHVI

# Analysis utilities, including a method to evaluate hypervolumes
from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax.modelbridge.torch import infer_objective_thresholds

# Import boar libraries
from boar.core.optimizer import BoarOptimizer # import the base class


class MooBOtorch(BoarOptimizer):
    # a class for multi-objective optimization
    def __init__(self,params = None, targets = None, warmstart = None, Path2OldXY = None, SaveOldXY2file = None, res_dir = 'temp', verbose = False) -> None:
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
            if par.optim_type == 'log':
                bounds = [np.log10(par.lims[0]), np.log10(par.lims[1])]
            else:
                bounds = [par.lims[0]/par.p0m, par.lims[1]/par.p0m] # remove the order of magnitude 


            Axparams.append({'name':par.name, 'type':'range', 'bounds':bounds})

        return Axparams
    
    def CreateSearchSpace(self, params):
        """Create the search space for the Ax/Botorch library

        Parameters
        ----------
        params : list of Fitparam() objects
            list of Fitparam() objects

        Returns
        -------
        SearchSpace() object
            SearchSpace() object

        """

        search_space_ = []
        for par in params:
            if par.relRange != 0:
                if par.optim_type == 'log':
                    # convert to log space
                    lower = np.log10(par.lims[0])
                    upper = np.log10(par.lims[1])
                else:
                    # remove the order of magnitude
                    lower = par.lims[0]/par.p0m
                    upper = par.lims[1]/par.p0m

                search_space_.append(RangeParameter(name=par.name, parameter_type=ParameterType.FLOAT, lower=lower, upper=upper))
            # else:
            #     if par.optim_type == 'log':
            #         # convert to log space
            #         value = np.log10(par.val)
            #     else:
            #         # remove the order of magnitude
            #         value = par.val
            #     search_space_.append(FixedParameter(name=par.name, parameter_type=ParameterType.FLOAT, value=par.p0m))

        return SearchSpace(parameters=search_space_)
            

    def obj_default(self, px, params, target, obj_type='MSE', loss='linear', threshold = 1000):

        #unpack the x and write values to params
        self.params_w(px, params) # write variable parameters into params, respecting user settings

        X = target['data']['X']
        y = target['data']['y']
        if 'weight' in target.keys(): 
            weight = target['weight']
        else:
            weight = 1
        yf = target['model'](X,params)

        z = self.obj_func_metric(target,yf,obj_type=obj_type)

        if 'loss' in target.keys(): # if loss is specified in targets, use it
            loss = target['loss']

        if threshold in target.keys(): # if threshold is specified in targets, use it
            threshold = target['threshold']

        z = self.lossfunc(z,loss,threshold)

        return z


    def ConvertTargets2Objectives(self, targets, obj_type='MSE', loss='linear', threshold = 1000):
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
        if not isinstance(threshold,list):
            threshold = threshold*np.ones(len(targets))
        # check if the length of the threshold list is the same as the number of metrics
        if len(threshold) != len(targets):
            raise ValueError('Threshold must either be a float or a list with the same length as the number of targets')

        AxObjectives,metrics = [],[]
        
        for _, target in enumerate(targets):
            params_names = [par.name for par in self.params if par.relRange != 0]
            metric_ = NoisyFunctionMetric(name=target['target_name'], param_names = params_names, noise_sd=0.0, lower_is_better=True)
            metric_.f = lambda px: self.obj_default(px, self.params, target, obj_type=obj_type, loss=loss, threshold = threshold[_])
            metrics.append(metric_)
            AxObjectives.append(Objective(metric=metric_))

        return AxObjectives,metrics

    def initialize_experiment(self, experiment,n_jobs_init, n_initial_points = 10):
        """Initialize the experiment

        Parameters
        ----------
        experiment : SimpleExperiment() object
            SimpleExperiment() object
        n_jobs_init : int
            number of parallel jobs
        n_initial_points : int, optional
            number of initial points, by default 10

        Returns
        -------

        """
        print('Initializing the experiment with {} initial points'.format(n_initial_points))
        sobol = Models.SOBOL(search_space=experiment.search_space, seed=1234)

        N_INIT = np.ceil(n_initial_points/n_jobs_init).astype(int)

        for _ in range(N_INIT):
            # experiment.new_trial(sobol.gen(n_jobs_init)).run()
            experiment.new_batch_trial(sobol.gen(n_jobs_init)).run()
            

        return experiment.fetch_data()
    
    def plot_all_objectives(self, outcomes):
        """Plot all objectives in a pairplot

        Parameters
        ----------
        outcomes : ndarray
            ndarray with shape (n,m) where n is the number of evaluations and m is the number of objectives

        Returns
        -------
        None

        """ 
        
        target_names = [target['target_name'] for target in self.targets]
        df  = pd.DataFrame(outcomes,columns=target_names)
        g = sns.PairGrid(df, diag_sharey=False, corner=True)
        g.map_lower(sns.scatterplot, s=15)

        # set all axes to be log scale
        for i in range(len(target_names)):
            if self.targets[i]['axis_type'] == 'log':
                g.axes[i,i].set_xscale('log')
                g.axes[i,i].set_yscale('log')


        # mask the upper triangle
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        g.map_upper(sns.scatterplot, s=15)
        #mask diagonal
        for i in range(len(target_names)):
            g.axes[i,i].set_visible(False)

        # Rotate x-axis labels
        for ax in g.axes[-1, :]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            #same for minorticks
            plt.setp(ax.xaxis.get_minorticklabels(), rotation=45)

        # display ticks out of the plot on all axes and add right and top axes with same limits as bottom and left
        for i in range(len(target_names)):
            for j in range(len(target_names)):
                if i > j: # only for lower triangle
                    ax = g.axes[i,j]
                    ax.tick_params(axis='both', which='both', direction='out', top=True, right=True)
                    # set top and right axis as visible
                    ax.spines['top'].set_visible(True)
                    ax.spines['right'].set_visible(True)

        
        plt.tight_layout

    def plot_density(self, experiment):
        """ Plot the density of the of the points in the search space

        Parameters
        ----------
        experiment : SimpleExperiment() object
            SimpleExperiment() object

        Raises
        ------
        ValueError
            axis_type must be either log or linear

        """        
        pnames = [p.name for p in self.params if p.relRange != 0]
        pnames_display = [p.display_name for p in self.params if p.relRange != 0]
        lims_low = [p.lims[0] for p in self.params if p.relRange != 0]
        lims_high = [p.lims[1] for p in self.params if p.relRange != 0]
        p_axis_type = [p.axis_type for p in self.params if p.relRange != 0]
        
 
        df = exp_to_df(experiment).sort_values(by=["trial_index"])
  
        df2 = df[pnames]

        # udpate values in df2 if optim_type is log
        for i in range(len(self.params)):
        # for i in range(len(pnames)):
            if self.params[i].relRange != 0:
                if self.params[i].optim_type == 'log':
                    df2[self.params[i].name] = 10**(df2[self.params[i].name]) # rescale to original value
                elif self.params[i].optim_type == 'linear':
                    df2[self.params[i].name] = df2[self.params[i].name]*self.params[i].p0m # rescale to original value
                else:
                    raise ValueError('axis_type must be either log or linear')
        

        # plt.figure()
        g = sns.PairGrid(df2,diag_sharey=False)
        # g.map_diag(sns.histplot,color='black',log_scale=False, kde=True)
        g.map_upper(sns.scatterplot, s=10, color=".3", marker="o")
        g.map_lower(sns.kdeplot, fill=True, thresh=0, levels=20, cmap='viridis',shade_lowest=False)
        g.map_lower(sns.scatterplot, s=10, color=".3", marker="o")

        for i in range(len(pnames)): # histogram on diagonal and kde on the diagonal
            if p_axis_type[i] == 'log':
                sns.histplot(color='black',log_scale=True, kde=True,ax = g.axes[i,i],data=df2[pnames[i]])
            elif p_axis_type[i] == 'linear':
                sns.histplot(color='black',log_scale=False, kde=True,ax = g.axes[i,i],data=df2[pnames[i]])
            else:
                raise ValueError('axis_type must be either log or linear')

        # set log axis is axis_type is log
        for i in range(len(pnames)):
            for j in range(len(pnames)):
                if j == 0:
                    g.axes[i,j].set_ylabel(pnames_display[i])
                    
                if i == len(pnames)-1:
                    g.axes[i,j].set_xlabel(pnames_display[j])

                if i > j: # only do the upper triangle
                    g.axes[i,j].set_yscale(p_axis_type[i])
                    g.axes[i,j].set_xscale(p_axis_type[j])
                    g.axes[i,j].set_ylim(lims_low[i],lims_high[i])
                    g.axes[i,j].set_xlim(lims_low[j],lims_high[j])

                elif i < j:
                    g.axes[i,j].set_xscale(p_axis_type[j])
                    g.axes[i,j].set_yscale(p_axis_type[i])
                    g.axes[i,j].set_xlim(lims_low[j],lims_high[j])
                    g.axes[i,j].set_ylim(lims_low[i],lims_high[i])


                else:
                    g.axes[i,j].set_xscale(p_axis_type[j])
                    g.axes[i,j].set_xlim(lims_low[j],lims_high[j])


        # Rotate x-axis labels
        for ax in g.axes[-1, :]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            #same for minorticks
            plt.setp(ax.xaxis.get_minorticklabels(), rotation=45)

        # display ticks out of the plot on all axes and add right and top axes with same limits as bottom and left
        for ax in g.axes.flat:
            ax.tick_params(axis='both', which='both', direction='out', top=True, right=True)
            # set top and right axis as visible
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)

        plt.tight_layout()

    def EvhiOptimizer(self,n_jobs=1, n_initial_points = 5, n_BO=10, n_jobs_init=None,obj_type='MSE',loss='linear',threshold=100):
        """Optimize the model using the Ax/Botorch library
        Uses the Expected Hypervolume Improvement (EHVI) algorithm

        Parameters
        ----------
        n_jobs : int, optional
            number of parallel jobs, by default 1
        n_initial_points : int, optional
            number of initial points, by default 5
        n_BO : int, optional
            number of Bayesian Optimization steps, by default 10
        n_jobs_init : int, optional
            number of parallel jobs for the initial points, by default None
        obj_type : str, optional
            the type of objective function to be used, by default 'MSE'
        loss : str, optional
            the loss function to be used, by default 'linear'
        threshold : float or list, optional
            the threshold used to calculate the hyper volume, by default 100

        Returns
        -------
        list of dict
            list of dictionaries with the following keys:\\
                'popt': ndarray with shape (m,) where m is the number of variable parameters that are optimized wi


        """    

        p0,lb,ub = self.params_r(self.params) # read out Fitparams & respect settings, NEEDED to updateb the self.params for p0m

        target_names = [target['target_name'] for target in self.targets]
        if n_jobs_init is None: # if n_jobs_init is not specified, use n_jobs
            n_jobs_init = n_jobs

        # if threshold is not a list, make it a list
        if not isinstance(threshold,list):
            threshold = threshold*np.ones(len(self.targets))
        # check if the length of the threshold list is the same as the number of metrics
        if len(threshold) != len(self.targets):
            raise ValueError('Threshold must either be a float or a list with the same length as the number of targets')

        search_space = self.CreateSearchSpace(self.params)

        objectives,metrics = self.ConvertTargets2Objectives(self.targets,obj_type=obj_type,loss=loss,threshold=threshold)
        mo = MultiObjective(objectives=objectives,)

        

        objective_thresholds = [ObjectiveThreshold(metric=metric, bound=val, relative=False)  for metric, val in zip(mo.metrics, threshold)]

        optimization_config = MultiObjectiveOptimizationConfig(objective = mo,objective_thresholds = objective_thresholds)

        # create the experiment
        ehvi_experiment = Experiment(name="MOO_EVHI", search_space=search_space, optimization_config=optimization_config, runner=SyntheticRunner(),)

        # initialize the experiment
        ehvi_data = self.initialize_experiment(ehvi_experiment,n_jobs_init, n_initial_points = n_initial_points)
        
        ehvi_hv_list = []
        ehvi_model = None
        N_BATCH = np.ceil(n_BO/n_jobs).astype(int)
        print(f'Running {N_BATCH} batches of {n_jobs} jobs each')
        tstart = time()
        for i in range(N_BATCH):   
            ehvi_model = get_MOO_NEHVI(
                experiment=ehvi_experiment, 
                data=ehvi_data,
            )
            generator_run = ehvi_model.gen(n_jobs)
            if n_jobs == 1:
                trial = ehvi_experiment.new_trial(generator_run=generator_run)
            else:
                trial = ehvi_experiment.new_batch_trial(generator_run=generator_run)
                
            
            trial.run()

            
            # with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            #     futures = []
            #     for i in range(n_jobs):
            #         futures.append(executor.submit(trial.run))

            #     for future in futures:
            #         future.result()
            

            ehvi_data = Data.from_multiple_data([ehvi_data, trial.fetch_data()])
            
            exp_df = exp_to_df(ehvi_experiment)

            outcomes = np.array(exp_df[target_names], dtype=np.double)
            try:
                hv = observed_hypervolume(modelbridge=ehvi_model)
            except:
                hv = 0
                print("Failed to compute hv")
            ehvi_hv_list.append(hv)
            print(f"Iteration: {i}, HV: {hv:.2e}")
        print(f'Finished in {time()-tstart} seconds')
        ehvi_outcomes = np.array(exp_to_df(ehvi_experiment)[target_names], dtype=np.double)
        
        # get the best parameters
        data =ehvi_experiment.fetch_data()
        df = data.df


        best_arm_name = df.arm_name[df['mean'] == df['mean'].min()].values[0]
        best_arm = ehvi_experiment.arms_by_name[best_arm_name].parameters
        best_arm_val = [best_arm[p.name] for p in self.params if p.relRange != 0]
        xmin = []
        std = 0*np.ones(len(best_arm_val))
        idx = 0
        self.params_w(best_arm_val,self.params,std=std)
        for par in self.params:
            if par.relRange != 0:
                if par.optim_type == 'log':
                    xmin.append(10**(best_arm_val[idx]))
                else:
                    xmin.append(best_arm_val[idx]*par.p0m)
                idx += 1    
        self.plot_all_objectives(ehvi_outcomes)
        self.plot_density(ehvi_experiment)
        return {'popt':xmin, 'data':df, 'experiment':ehvi_experiment, 'model':ehvi_model, 'outcomes':ehvi_outcomes, 'hv':ehvi_hv_list, 'metric':metrics}


    
