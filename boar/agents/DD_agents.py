######################################################################
#################### Drift diffusion agent ###########################
######################################################################
# Version 0.1
# (c) Larry Lueer, Vincent M. Le Corre, i-MEET 2021-2023

# Import libraries
import os,itertools,warnings,sys,copy
from scipy import interpolate

# Import boar
from boar import *
from boar.agents.Agent import Agent
# Import homemade package by VLC
from boar.SIMsalabim_utils.RunSim import *
from boar.SIMsalabim_utils.CalcFom import *
from boar.SIMsalabim_utils.GetInputPar import *
from boar.SIMsalabim_utils.aux_func import *


class Drift_diffusion_agent(Agent):
    """ Agent to run drift diffusion simulations with SIMsalabim to be used with BOAR MultiObjectiveOptimizer
    
    Parameters
    ----------
    path2simu : str
        Path to the simulation executable

    """
    def __init__(self,path2simu = '',SafeModeParallel = True) -> None:
        super().__init__()
        self.path2simu = path2simu
        self.SafeModeParallel = SafeModeParallel
    

    def DriftDiffusion(self,X,params,X_dimensions=[],max_jobs=3,fixed_str=''):
        """ Run the drift diffusion simulations for a given list of parameters 

        Parameters
        ----------
        X : np.array
            Array of fixed parameters (like Voltages, light intensities, etc.)) 
        params : list
            list of Fitparam objects
        X_dimensions : list, optional
            name of the fixed parameters in X, by default []
        max_jobs : int, optional
            maximum number of jobs to run in parallel, by default 3
        fixed_str : str, optional
            string of fixed parameters to be passed to SIMsalabim, by default ''

        Returns
        -------
        np.array
            Array of containing the simulation results (Current Density)
        """        
        

        fixed_params = {} # dict of structure: {X_dimension:ordered list of unique params}


        # Get the unique values of the experimental parameters
        X_dimx = [XX for XX in X_dimensions if XX!='Vext'] # exclude Vext from the list of experimental parameters
        idimX = [ii for ii,XX in enumerate(X_dimensions) if XX!='Vext'] # get the indices of the experimental parameters without Vext
        
        LU = 1 # need to initialize LU
        for idim,xdim in zip(idimX,X_dimx):
            uniques1,iau = np.unique(X[:,idim],return_index=True) # make sure we don't drop the index of the first occurence of a unique value
            iaus = np.argsort(iau) # sort the indices of the first occurence of a unique value
            uniques = uniques1[iaus] # sort the unique values
            fixed_params[X_dimensions[idim]] = uniques  
            LU = len(uniques) # this should be the same for all          
        
        str_lst,labels,JV_files,Var_files,path_lst,code_name_lst,scPars_files,val,nam,V,JV_files_all = [],[],[],[],[],[],[],[],[],[],[]

        for param in params:
            val.append([param.val]) # param vals in order of param
            nam.append(param.name) # names in order of param


        Y = []
        Xs = []
        
        exptl_param_names = [k for k,v in fixed_params.items()]
        nam = nam + exptl_param_names
        for ii in range(LU):
            exptl_params = [v[ii] for k,v in fixed_params.items()]
            
            i = [vv[0] for vv in val] + exptl_params
            str_line = fixed_str + ' '
            lab = ''
            JV_name = 'JV'
            Var_name = 'none'
            short_dic = {}
            scPars_name = 'scPars'
            dum = []
            for j,name in zip(i,nam):
                str_line = str_line +'-'+name+' {:.3e} '.format(j)
                JV_name = JV_name +'_'+name +'_{:.3e}'.format(j)
                scPars_name = scPars_name +'_'+ name +'_{:.3e}'.format(j)
                if name in X_dimensions:
                    if name != 'Vext':
                        dum.append(j)
                        
            Xs.append(dum)
            

            if self.SafeModeParallel:
                JV_name = 'JV_' + str(uuid.uuid4())
                str2run =  str_line+ '-JV_file '+JV_name+ '.dat -Var_file none'
                str_lst.append(str2run)
                JV_files.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                JV_files_all.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                Var_files.append(os.path.join(self.path2simu , str(Var_name+ '.dat')))
                scPars_files.append(os.path.join(self.path2simu , str(scPars_name+ '.dat')))
                code_name_lst.append('simss')
                path_lst.append(self.path2simu)
                labels.append(lab)
            else:
                str2run =  str_line+ '-JV_file '+JV_name+ '.dat -Var_file none'
                str_lst.append(str2run)
                JV_files.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                JV_files_all.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                Var_files.append(os.path.join(self.path2simu , str(Var_name+ '.dat')))
                scPars_files.append(os.path.join(self.path2simu , str(scPars_name+ '.dat')))
                code_name_lst.append('simss')
                path_lst.append(self.path2simu)
                labels.append(lab)

        Simulation_Input = str_lst,JV_files,Var_files,scPars_files,code_name_lst,path_lst,labels
        RunSimulation(Simulation_Input,max_jobs=max_jobs)

        for JV_name,fix in zip(JV_files_all,Xs):

            volt = []
            for i in X:
                if (i[1:] == np.asarray(fix)).all():
                    volt.append(i[0])

            # Read JV file output from SIMsalabim
            # Read JV file output from SIMsalabim
            try:
                df = pd.read_csv(os.path.join(JV_name),delim_whitespace=True)
            except:
                raise Exception('Could not read JV file: ',JV_name)


            if min(volt) < min(df['Vext']):
                warnings.warn('Vmin of the experimental data is outside the range of the simulated data. This may lead to incorrect results. Please decrease Vmin in SIMsalabim.', UserWarning)  
            if max(volt) > max(df['Vext']):
                warnings.warn('Vmax of the experimental data is outside the range of the simulated data. This may lead to incorrect results. Please increase Vmax in SIMsalabim.', UserWarning)  

            # Do interpolation in case SIMsalabim did not return the same number of points
            try: # try sline interpolation first and if it fails use simple linear interpolation
                tck = interpolate.splrep(df['Vext'], df['Jext'], s=0)
                Ysin = interpolate.splev(volt, tck, der=0)
            except:
                f = interpolate.interp1d(df['Vext'], df['Jext'],kind='linear')
                Ysin = f(volt)

            if len(Y) == 0:
                Y = Ysin
            else:
                Y = np.append(Y,Ysin)

        return Y


    def get_FOM(self, params_dict, FOMs):
        """ Calculates the FOMs for a given set of parameters 

        Parameters
        ----------
        params_dict : dict
            Dictionary of the parameters
        FOMs : list
            list of FOMparam with the FOMs to be calculated
        
        Returns
        -------
        dict
            Dictionary of the FOMs
        """      
        
        ParFileDic = ReadParameterFile(os.path.join(self.path2simu,'device_parameters.txt'))
        r = {}

        for fom in FOMs:
            if fom.optim_type == 'log':
                r.update({fom.name:np.log10(fom.func(params_dict,ParFileDic))})
            else:
                r.update({fom.name:fom.func(params_dict,ParFileDic)})

        return r

    
    def get_FOMs(self,X,params,FOMs):      
        """ Calculates the FOMs for a given list of parameters and returns a np.array of FOMs for each parameter set 

        Parameters
        ----------
        X : list
            list of lists of parameters used to calculate FOMs
        params : list
            list of Fitparam objects 
        FOMs : list
            list of FOMparam with the FOMs to be calculated
        
        Returns
        -------
        np.array
            Array of FOMs
        """        
        
        dev_params = ReadParameterFile(os.path.join(self.path2simu,'device_parameters.txt'))
        rows=[]
        for XX in X:
            row = []
            fit_params = {}
            jj=0
            for pp in params:
                if pp.relRange!=0:
                    v = XX[jj]
                    if pp.optim_type == 'log':
                        fit_params.update({pp.name:10**v})
                    else:
                        fit_params.update({pp.name:v* pp.p0m})
                    jj+=1
                else:
                    fit_params.update({pp.name:pp.val})
            
            for fom in FOMs:
                if fom.optim_type == 'log':
                    row.append(np.log10(fom.func(fit_params,dev_params)))
                else:
                    row.append(fom.func(fit_params,dev_params))

            rows.append(row)
        
        rows = np.array(rows)
        return(rows)


    def return_FOM(self, fit_params, FOMs, True_params={}):
        """ Print the FOMs for a given set of parameters 
        if True_params is given, also prints the fit and true values

        Parameters
        ----------
        fit_params : dict
            dictionary of fit parameters
        FOMs : list
            list of FOMparam with the FOMs to be calculated
        True_params : dict, optional
            dictionary of true parameters, by default {}
        """        
        ParFileDic = ReadParameterFile(os.path.join(self.path2simu,'device_parameters.txt'))


        print('\nFOMs:')
        for fom in FOMs:
            fit_params.update({fom.name:fom.func(fit_params,ParFileDic)})
            str2display = fom.name + ' Fitted value = ' + '{:.3e}'.format(fit_params[fom.name])
            if True_params != {}:
                True_params.update({fom.name:fom.func(True_params,ParFileDic)})
                str2display = str2display + ' True value = ' + '{:.3e}'.format(True_params[fom.name])
            
            print(str2display)
            
    def Compare_JVs_simu(self,target,params,params_true,FOMs,DD_func=None,verbose=False):
        """Compare the simulated JV with the result from the fit

        Parameters
        ----------
        target : dict
            Dictionary with the target parameters
        params :list
            list of Fitparam objects with the fitted parameters
        params_true : list
            list of Fitparam objects with the true parameters
        FOMs : list
            list of FOMparam objects
        
        """ 
        if DD_func == None:
            DD_func = self.DriftDiffusion

        x_dims = target['data']['X_dimensions']   
        x_units = target['data']['X_units'] 
        y_dim = target['data']['y_dimension']  
        y_unit = target['data']['y_unit']    

        xnames = [x_dim + f' [{x_unit}]' for x_dim,x_unit in zip(x_dims,x_units)]
        xnames[0] = 'V [V]'
        yname = y_dim + f' [{y_unit}]'

        Xplot = np.unique(target['data']['X'][:,1:])
        X_dimensions = target['data']['X_dimensions']
        if 'V' in X_dimensions and 'Vext' not in X_dimensions: # make sure that Vext is here
            index = X_dimensions.index('V')
            X_dimensions[index]= 'Vext'
        X = target['data']['X']
        
        # Print the results and calculate FOMs
        fit_params,True_params = {},{}
        strfit = ''
        strtrue = ''
        for i,j in zip(params,params_true):
            fit_params[i.name] = i.val

            strfit += '-'+i.name+' '+str(i.val)+' '

            True_params[j.name] = j.val
            strtrue += '-'+j.name+' '+str(j.val)+' '

        if verbose:
            # print str to be used by SIMsalabim
            print(strfit)
            print(strtrue)
            # print the figures of merit
            if FOMs != []:
                self.return_FOM(fit_params,FOMs,True_params=True_params)


        # Rerun the model with true and fitted values
        ytrue = DD_func(target['data']['X'],params_true,X_dimensions=X_dimensions)
        yfit = DD_func(target['data']['X'],params,X_dimensions=X_dimensions)

        # plot the result 
        f = plt.figure(1,figsize=(16,12))
        idx = 0
        units = x_units[1:]
        
        
        for i in Xplot:
            lab = ''

            if len(X_dimensions)==2:
                lab = str(X_dimensions[1]) + ' = ' + str(i) + ' ' + units[0] + ' '
            else:
                for ii, name in enumerate(X_dimensions[1:]):
                    lab = lab + name + ' = ' + str(i[ii]) + ' ' + units[ii] + ' '
            pick = X[:,1:]==i
            pick2 = []
            for i in pick:
                if i.all() == True:
                    pick2.append(True)
                else:
                    pick2.append(False)

            xp = X[pick2,0]
            yp = ytrue[pick2]
            yf = yfit[pick2]
            
            plt.plot(xp,yp,'o',c='C'+str(idx),label=lab)
            plt.plot(xp,yf,c='C'+str(idx))
            idx += 1
        plt.xlabel(xnames[0])
        plt.ylabel(yname)
        plt.legend(loc='best')
        plt.show()


    def Compare_JVs_exp(self,target,params,FOMs=None,xlim=[],ylim=[],fixed_str='',DD_func=None,verbose=False,kwargs=None):
        """Compare the simulated JV with the result from the fit

        Parameters
        ----------
        target : dict
            Dictionary with the target parameters
        params :list
            list of Fitparam objects with the fitted parameters
        FOMs : list, optional
            list of FOMparam objects, by default None
        xlim : list, optional
            list of 2 elements with the x limits, by default []
        ylim : list, optional
            list of 2 elements with the y limits, by default []
        fixed_str : str, optional
            string with the fixed parameters, by default ''
        verbose : bool, optional
            print the results, by default False
        kwargs : dict, optional
            dictionary with the plotting options, by default None

                including:
                    savefig : bool, optional
                        save the figure, by default True
                    figname : str, optional
                        name of the figure, by default 'JV_fit'
                    figdir : str, optional
                        directory where to save the figure, by default ''
                    figext : str, optional
                        extension of the figure, by default '.png'
                    figsize : tuple, optional
                        size of the figure, by default (16,12)
                    figdpi : int, optional
                        dpi of the figure, by default 300
        """  
        if DD_func == None:
            DD_func = self.DriftDiffusion
        # if kwargs != {}:
        if kwargs is None:
            kwargs = {}
        show_fig = kwargs.get('show_fig',True)
        clear_output = kwargs.get('clear_output',True)
        savefig = kwargs.get('savefig',True)
        show_fig = kwargs.get('show_fig',True)
        figname = kwargs.get('figname','JV_fit')
        figdir = kwargs.get('figdir','')
        figext = kwargs.get('figext','.png')
        figsize = kwargs.get('figsize',(10,8))
        figdpi = kwargs.get('figdpi',300) 

        

        x_dims = target['data']['X_dimensions']   
        x_units = target['data']['X_units'] 
        y_dim = target['data']['y_dimension']  
        y_unit = target['data']['y_unit']    

        xnames = [x_dim + f' [{x_unit}]' for x_dim,x_unit in zip(x_dims,x_units)]
        xnames[0] = 'V [V]'
        yname = y_dim + f' [{y_unit}]'

        Xplot = np.unique(target['data']['X'][:,1:])
        X_dimensions = target['data']['X_dimensions']
        if 'V' in X_dimensions and 'Vext' not in X_dimensions: # make sure that Vext is here
            index = X_dimensions.index('V')
            X_dimensions[index]= 'Vext'
        X = target['data']['X']
        
        # Print the results and calculate FOMs
        fit_params,True_params = {},{}
        strfit = ''
        strtrue = ''
        for i in params:
            fit_params[i.name] = i.val

            strfit += '-'+i.name+' '+str(i.val)+' '


        if verbose:
            # print str to be used by SIMsalabim
            print(strfit)

            # print the figures of merit
            if FOMs is not None:
                self.return_FOM(fit_params,FOMs)


        # Rerun the model with true and fitted values
        ytrue = target['data']['y']

        
        yfit = DD_func(target['data']['X'],params,X_dimensions=X_dimensions,fixed_str=' '+fixed_str)
        

        # plot the result 
        f = plt.figure(99,figsize=figsize)

        if clear_output:
            # refresh the plot
            f.clear()

        # plt.figure(figsize=(16,12))
        idx = 0
        units = x_units[1:]
        
        
        for i in Xplot:
            lab = ''

            if len(X_dimensions)==2:
                lab = str(X_dimensions[1]) + ' = ' + str(i) + ' ' + units[0] + ' '
            else:
                for ii, name in enumerate(X_dimensions[1:]):
                    lab = lab + name + ' = ' + str(i[ii]) + ' ' + units[ii] + ' '
            pick = X[:,1:]==i
            pick2 = []
            for i in pick:
                if i.all() == True:
                    pick2.append(True)
                else:
                    pick2.append(False)

            xp = X[pick2,0]
            yp = ytrue[pick2]
            yf = yfit[pick2]
            
            plt.plot(xp,yp,'o',c='C'+str(idx),label=lab)
            plt.plot(xp,yf,c='C'+str(idx))
            # plt.loglog(abs(xp),abs(yp),'o',c='C'+str(idx),label=lab)
            # plt.loglog(abs(xp),abs(yf),c='C'+str(idx))
            idx += 1

        if xlim!=[]:
            plt.xlim(xlim)
        if ylim!=[]:
            plt.ylim(ylim)
        plt.xlabel(xnames[0])
        plt.ylabel(yname)
        plt.legend(loc='best')
        if show_fig:
            plt.show()

        if savefig:
            f.savefig(os.path.join(figdir,figname+figext), dpi=300)

       # return f


    def DriftDiffusion_relative(self,X,params,X_dimensions=[],max_jobs=3,fixed_str='',dev_par_fname='',**kwargs):
        """ Run the drift diffusion simulations for a given list of parameters\\
        Use the relative energy level positions, see Check_fit_params for more info.

        Parameters
        ----------
        X : np.array
            Array of fixed parameters (like Voltages, light intensities, etc.)) 
        params : list
            list of Fitparam objects
        X_dimensions : list, optional
            name of the fixed parameters in X, by default []
        max_jobs : int, optional
            maximum number of jobs to run in parallel, by default 3
        fixed_str : str, optional
            string of fixed parameters to be passed to SIMsalabim, by default ''
        dev_par_fname : str, optional
            can be used to update the name of the file containing the device parameters be careful not to provide the dev_par filename in fixed_str, by default ''
        **kwargs : dict, optional
            additional arguments to be passed to the DriftDiffusion function, by default {}
                including:
                    ignore_error_code : bool, optional
                        if True, ignore the error code returned by SIMsalabim, by default True
        Returns
        -------
        np.array
            Array of containing the simulation results (Current Density)
        """  
        ignore_error_code = kwargs.get('ignore_error_code',True)
        # print('Running DriftDiffusion_relative')
        if dev_par_fname=='': # if no dev_par_fname is provided, use the default one
            dev_par_fname = ''
            ParFileDic = ReadParameterFile(os.path.join(self.path2simu,'device_parameters.txt'))
        else:
            ParFileDic = ReadParameterFile(os.path.join(self.path2simu,dev_par_fname))
            fixed_str = dev_par_fname + ' ' + fixed_str # add the dev_par_fname at the beginning of the fixed_str

        fixed_params = {} # dict of structure: {X_dimension:ordered list of unique params}


        # Get the unique values of the experimental parameters
        X_dimx = [XX for XX in X_dimensions if XX!='Vext'] # exclude Vext from the list of experimental parameters
        idimX = [ii for ii,XX in enumerate(X_dimensions) if XX!='Vext'] # get the indices of the experimental parameters without Vext
        
        LU = 1 # need to initialize LU
        for idim,xdim in zip(idimX,X_dimx):
            uniques1,iau = np.unique(X[:,idim],return_index=True) # make sure we don't drop the index of the first occurence of a unique value
            iaus = np.argsort(iau) # sort the indices of the first occurence of a unique value
            uniques = uniques1[iaus] # sort the unique values
            fixed_params[X_dimensions[idim]] = uniques  
            LU = len(uniques) # this should be the same for all          
        
        str_lst,labels,JV_files,Var_files,path_lst,code_name_lst,scPars_files,val,nam,V,JV_files_all = [],[],[],[],[],[],[],[],[],[],[]

        for param in params:
            val.append([param.val]) # param vals in order of param
            nam.append(param.name) # names in order of param

        Y = []
        Xs = []

        
        exptl_param_names = [k for k,v in fixed_params.items()]
        nam = nam + exptl_param_names
        for ii in range(LU):
            exptl_params = [v[ii] for k,v in fixed_params.items()]
            
            i = [vv[0] for vv in val] + exptl_params
            

            str_line = fixed_str + ' '
            lab = ''
            JV_name = 'JV'
            Var_name = 'none'
            short_dic = {}
            scPars_name = 'scPars'
            dum = []

            # correct the parameters for the relative energy level positions
            CorrectedParams = self.Check_fit_params(i,nam,ParFileDic)
            
            # create the string to be passed to SIMsalabim
            for j,name in zip(i,nam):
                if name in CorrectedParams.keys():
                    dumval = CorrectedParams[name]
                else:
                    dumval = j
                str_line = str_line +'-'+name+' {:.3e} '.format(dumval)
                JV_name = JV_name +'_'+name +'_{:.3e}'.format(dumval)
                scPars_name = scPars_name +'_'+ name +'_{:.3e}'.format(dumval)
                if name in X_dimensions:
                    if name != 'Vext':
                        dum.append(j)
                        
            Xs.append(dum)
            

            if self.SafeModeParallel:
                JV_name = 'JV_' + str(uuid.uuid4())
                str2run =  str_line+ '-JV_file '+JV_name+ '.dat -Var_file none'
                str_lst.append(str2run)
                JV_files.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                JV_files_all.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                Var_files.append(os.path.join(self.path2simu , str(Var_name+ '.dat')))
                scPars_files.append(os.path.join(self.path2simu , str(scPars_name+ '.dat')))
                code_name_lst.append('simss')
                path_lst.append(self.path2simu)
                labels.append(lab)
            else:
                str2run =  str_line+ '-JV_file '+JV_name+ '.dat -Var_file none'
                str_lst.append(str2run)
                JV_files.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                JV_files_all.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                Var_files.append(os.path.join(self.path2simu , str(Var_name+ '.dat')))
                scPars_files.append(os.path.join(self.path2simu , str(scPars_name+ '.dat')))
                code_name_lst.append('simss')
                path_lst.append(self.path2simu)
                labels.append(lab)
        
        Simulation_Input = str_lst,JV_files,Var_files,scPars_files,code_name_lst,path_lst,labels

        RunSimulation(Simulation_Input,max_jobs=max_jobs,do_multiprocessing=True,ignore_error_code=ignore_error_code)

        idx = 0
        for JV_name,fix in zip(JV_files_all,Xs):
            volt = []
            for i in X:
                if (i[1:] == np.asarray(fix)).all():
                    volt.append(i[0])

            # Read JV file output from SIMsalabim
            try:
                df = pd.read_csv(JV_name,delim_whitespace=True)
            except Exception as e:
                print(str_lst[idx])
                raise Exception('Could not read JV file: ',JV_name)
            
            if len(df) <= 3:
                print(str_lst[idx])
                raise Exception('The JV file is empty. Please check the input parameters.')
            # Do interpolation in case SIMsalabim did not return the same number of points
            try: # try sline interpolation first and if it fails use simple linear interpolation
                tck = interpolate.splrep(df['Vext'], df['Jext'], s=0)
                Ysin = interpolate.splev(volt, tck, der=0,ext = 0)
            except Exception as e:
                # try linear interpolation
                try:  
                    f = interpolate.interp1d(df['Vext'], df['Jext'],kind='linear',fill_value='extrapolate')
                    Ysin = f(volt)
                except Exception as e2: # if linear interpolation fails, use the minimum value of the JV curve
                    if min(volt)- 0.025 < min(df['Vext']): # need this as a safety to make sure we output something
                        # add a point at the beginning of the JV curve
                        df = df.append({'Vext':min(volt),'Jext':df['Jext'].iloc[0]},ignore_index=True)
                        df = df.sort_values(by=['Vext'])
                    f = interpolate.interp1d(df['Vext'], df['Jext'],kind='linear',fill_value='extrapolate')
                    Ysin = f(volt)


            if len(Y) == 0:
                Y = Ysin
            else:
                Y = np.append(Y,Ysin)
            
            # Delete the JV file
            os.remove(JV_name)
            idx += 1

                

        return Y
    
    def DriftDiffusion_rela_nrj_Trap_dist(self,X,params,X_dimensions=[],max_jobs=3,fixed_str='',dev_par_fname='',Trap_dist=None,test=True,**kwargs):
        """ Run the drift diffusion simulations for a given list of parameters\\
        Use the relative energy level positions, see Check_fit_params for more info.
        Also, input the custom trap distribution with different models.
        For now, only the urbach tails and urbach tail plus mid gap traps are implemented.

        Parameters
        ----------
        X : np.array
            Array of fixed parameters (like Voltages, light intensities, etc.)) 
        params : list
            list of Fitparam objects
        X_dimensions : list, optional
            name of the fixed parameters in X, by default []
        max_jobs : int, optional
            maximum number of jobs to run in parallel, by default 3
        fixed_str : str, optional
            string of fixed parameters to be passed to SIMsalabim, by default ''
        dev_par_fname : str, optional
            can be used to update the name of the file containing the device parameters be careful not to provide the dev_par filename in fixed_str, by default ''
        Trap_dist : str, optional
            name of the trap model to be used, by default None
                Can be:
                    'conduction_urbach' : urbach tails below the conduction band
                    'calence_urbach' : urbach tails above the valence band
                    'double_urbach' : urbach tails above the valence band and below the conduction band
                    'double_urbach_midgap' : urbach tails above the valence band and below the conduction band plus mid gap traps
                    None : no trap distribution is used (default)
        test : bool, optional
            if True, run a test simulation that does not transform the input, by default True
        **kwargs : dict, optional
            dictionary of optional parameters to be passed to the simulation code, by default {}
                including:
                    num_points : int, optional
                        number of points in the trap distribution, by default 20
                    ignore_error_code : bool, optional
                        if True, ignore the error code returned by SIMsalabim, by default True

        Returns
        -------
        np.array
            Array of containing the simulation results (Current Density)
        """  

        num_points = kwargs.get('num_points',20) # number of points in the trap distribution
        ignore_error_code = kwargs.get('ignore_error_code',True) # if True, ignore the error code returned by SIMsalabim

        if Trap_dist is None: # if no trap model is provided, use the default one
            self.DriftDiffusion_relative(X,params,X_dimensions=X_dimensions,max_jobs=max_jobs,fixed_str=fixed_str,dev_par_fname=dev_par_fname)  

        else:
            if dev_par_fname=='': # if no dev_par_fname is provided, use the default one
                dev_par_fname = ''
                ParFileDic = ReadParameterFile(os.path.join(self.path2simu,'device_parameters.txt'))
            else:
                ParFileDic = ReadParameterFile(os.path.join(self.path2simu,dev_par_fname))
                fixed_str = dev_par_fname + ' ' + fixed_str # add the dev_par_fname at the beginning of the fixed_str

            fixed_params = {} # dict of structure: {X_dimension:ordered list of unique params}

            # Get the unique values of the experimental parameters
            X_dimx = [XX for XX in X_dimensions if XX!='Vext'] # exclude Vext from the list of experimental parameters
            idimX = [ii for ii,XX in enumerate(X_dimensions) if XX!='Vext'] # get the indices of the experimental parameters without Vext
            
            LU = 1 # need to initialize LU
            for idim,xdim in zip(idimX,X_dimx):
                uniques1,iau = np.unique(X[:,idim],return_index=True) # make sure we don't drop the index of the first occurence of a unique value
                iaus = np.argsort(iau) # sort the indices of the first occurence of a unique value
                uniques = uniques1[iaus] # sort the unique values
                fixed_params[X_dimensions[idim]] = uniques  
                LU = len(uniques) # this should be the same for all          
            
            str_lst,labels,JV_files,Var_files,path_lst,code_name_lst,scPars_files,val,nam,V,JV_files_all = [],[],[],[],[],[],[],[],[],[],[]

            for param in params:
                val.append([param.val]) # param vals in order of param
                nam.append(param.name) # names in order of param

            # check if 'Eu' is in the list of parameters
            if 'Eu' in nam or 'Eu' in fixed_params.keys():
                if 'Eu' in nam:
                    idx = nam.index('Eu')
                    Eu = val[idx][0]
                elif 'Eu' in fixed_params.keys():
                    Eu = fixed_params['Eu'][0]
                else:
                    raise ValueError('Eu is not in the list of parameters, please add it to the list of parameters')
                    
                if Trap_dist == 'double_urbach_midgap':
                    if 'frac_mid' in nam or 'frac_mid' in fixed_params.keys():
                        if 'frac_mid' in nam:
                            idx = nam.index('frac_mid')
                            frac_mid = val[idx][0]
                        elif 'frac_mid' in fixed_params.keys():
                            frac_mid = fixed_params['frac_mid'][0]
                        else:
                            raise ValueError('frac_mid is not in the list of parameters, please add it to the list of parameters')
                        if frac_mid < 0 or frac_mid > 1:
                            raise ValueError('frac_mid must be between 0 and 1')
                    else:
                        raise ValueError('frac_mid is not in the list of parameters, please add it to the list of parameters')
            else:
                raise ValueError('Eu is not in the list of parameters, please add it to the list of parameters')
                
            if num_points <= 0:
                raise ValueError('num_points must be greater than 0')
            
            num_points = int(num_points) # make sure num_points is an integer

            # create the trap distribution file for SIMsalabim
            CB = ParFileDic['CB'] # conduction band default value
            VB = ParFileDic['VB'] # valence band default value
            if 'CB' in fixed_params.keys(): # if CB is in the list of fixed parameters, use the value of CB
                CB = fixed_params['CB'][0]
            elif 'CB' in nam: # if CB is in the list of parameters, use the value of CB
                idx = nam.index('CB')
                CB = val[idx][0]
            if 'VB' in fixed_params.keys(): # if VB is in the list of fixed parameters, use the value of VB
                VB = fixed_params['VB'][0]
            elif 'VB' in nam: # if VB is in the list of parameters, use the value of VB
                idx = nam.index('VB')
                VB = val[idx][0]

            # make sure CB and VB are floats
            CB = float(CB)
            VB = float(VB)

            if Trap_dist == 'conduction_urbach':
                E,Ntraps = conduction_urbach(CB,VB,Eu,num_points)
            elif Trap_dist == 'valence_urbach':
                E,Ntraps = valence_urbach(CB,VB,Eu,num_points)
            elif Trap_dist == 'double_urbach_midgap':
                E,Ntraps = double_urbach_midgap(CB,VB,Eu,frac_mid,num_points)
            elif Trap_dist == 'double_urbach':
                E,Ntraps = double_urbach(CB,VB,Eu,num_points)
            else:
                raise ValueError('Trap_dist must be ''conduction_urbach'', ''valence_urbach'', ''double_urbach_midgap'' or ''double_urbach''')

            # create trap file
            df_trap = pd.DataFrame({'E':E,'Ntraps':Ntraps})
            filename = f'urbach_{Eu*1e3:.3e}_mV.txt'
            # set numerb of digits to 4
            df_trap.to_csv(os.path.join(self.path2simu,filename),sep='\t',index=False,header=True,float_format='%.4e')

            # append the trap file to fixed_str
            fixed_str = fixed_str + ' -BulkTrapFile ' + filename

            Y = []
            Xs = []

            
            exptl_param_names = [k for k,v in fixed_params.items()]
            nam = nam + exptl_param_names
            for ii in range(LU):
                exptl_params = [v[ii] for k,v in fixed_params.items()]
                
                i = [vv[0] for vv in val] + exptl_params
                

                str_line = fixed_str + ' '
                lab = ''
                JV_name = 'JV'
                Var_name = 'none'
                short_dic = {}
                scPars_name = 'scPars'
                dum = []

                # correct the parameters for the relative energy level positions
                CorrectedParams = self.Check_fit_params(i,nam,ParFileDic)
                
                # create the string to be passed to SIMsalabim
                for j,name in zip(i,nam):
                    
                    if name in CorrectedParams.keys():
                        dumval = CorrectedParams[name]
                    else:
                        dumval = j

                    if name in ParFileDic.keys(): # don't pass the parameters that are in the ParFileDic to SIMsalabim, but keeo them for the labels
                        str_line = str_line +'-'+name+' {:.3e} '.format(dumval)

                    JV_name = JV_name +'_'+name +'_{:.3e}'.format(dumval)
                    scPars_name = scPars_name +'_'+ name +'_{:.3e}'.format(dumval)
                    if name in X_dimensions:
                        if name != 'Vext':
                            dum.append(j)
                            
                Xs.append(dum)
                

                if self.SafeModeParallel:
                    JV_name = 'JV_' + str(uuid.uuid4())
                    str2run =  str_line+ '-JV_file '+JV_name+ '.dat -Var_file none'
                    str_lst.append(str2run)
                    JV_files.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                    JV_files_all.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                    Var_files.append(os.path.join(self.path2simu , str(Var_name+ '.dat')))
                    scPars_files.append(os.path.join(self.path2simu , str(scPars_name+ '.dat')))
                    code_name_lst.append('simss')
                    path_lst.append(self.path2simu)
                    labels.append(lab)
                else:
                    str2run =  str_line+ '-JV_file '+JV_name+ '.dat -Var_file none'
                    str_lst.append(str2run)
                    JV_files.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                    JV_files_all.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                    Var_files.append(os.path.join(self.path2simu , str(Var_name+ '.dat')))
                    scPars_files.append(os.path.join(self.path2simu , str(scPars_name+ '.dat')))
                    code_name_lst.append('simss')
                    path_lst.append(self.path2simu)
                    labels.append(lab)
            
            Simulation_Input = str_lst,JV_files,Var_files,scPars_files,code_name_lst,path_lst,labels

            RunSimulation(Simulation_Input,max_jobs=max_jobs,do_multiprocessing=True,ignore_error_code=ignore_error_code)

            idx = 0
            for JV_name,fix in zip(JV_files_all,Xs):
                volt = []
                for i in X:
                    if (i[1:] == np.asarray(fix)).all():
                        volt.append(i[0])

                # Read JV file output from SIMsalabim
                try:
                    df = pd.read_csv(JV_name,delim_whitespace=True)
                except:
                    print(str_lst[idx])
                    raise Exception('Could not read JV file: ',JV_name)
                
                # Do interpolation in case SIMsalabim did not return the same number of points
                try: # try sline interpolation first and if it fails use simple linear interpolation
                    tck = interpolate.splrep(df['Vext'], df['Jext'], s=0)
                    Ysin = interpolate.splev(volt, tck, der=0,ext = 0)
                except Exception as e:
                    # try linear interpolation
                    try:  
                        f = interpolate.interp1d(df['Vext'], df['Jext'],kind='linear',fill_value='extrapolate')
                        Ysin = f(volt)
                    except: # if linear interpolation fails, use the minimum value of the JV curve
                        if min(volt)- 0.025 < min(df['Vext']): # need this as a safety to make sure we output something
                            # add a point at the beginning of the JV curve
                            df = df.append({'Vext':min(volt),'Jext':df['Jext'].iloc[0]},ignore_index=True)
                            df = df.sort_values(by=['Vext'])
                        f = interpolate.interp1d(df['Vext'], df['Jext'],kind='linear',fill_value='extrapolate')
                        Ysin = f(volt)


                if len(Y) == 0:
                    Y = Ysin
                else:
                    Y = np.append(Y,Ysin)
                
                # Delete the JV file
                os.remove(JV_name)
                idx += 1


        return Y
    
    def DriftDiffusion4fit(self,X,params,X_dimensions=[],max_jobs=3,fixed_str='',dev_par_fname='',Trap_dist=None,test=True,**kwargs):
        """ Run the drift diffusion simulations for a given list of parameters\\
        Use the relative energy level positions, see Check_fit_params for more info.
        Also, input the custom trap distribution with different models.
        For now, only the urbach tails and urbach tail plus mid gap traps are implemented.
        Can also control the CNI, CPI ions densities being equal by using Cions as a parameter name.

        Parameters
        ----------
        X : np.array
            Array of fixed parameters (like Voltages, light intensities, etc.)) 
        params : list
            list of Fitparam objects
        X_dimensions : list, optional
            name of the fixed parameters in X, by default []
        max_jobs : int, optional
            maximum number of jobs to run in parallel, by default 3
        fixed_str : str, optional
            string of fixed parameters to be passed to SIMsalabim, by default ''
        dev_par_fname : str, optional
            can be used to update the name of the file containing the device parameters be careful not to provide the dev_par filename in fixed_str, by default ''
        Trap_dist : str, optional
            name of the trap model to be used, by default None
                Can be:
                    'conduction_urbach' : urbach tails below the conduction band
                    'calence_urbach' : urbach tails above the valence band
                    'double_urbach' : urbach tails above the valence band and below the conduction band
                    'double_urbach_midgap' : urbach tails above the valence band and below the conduction band plus mid gap traps
                    None : no trap distribution is used (default)
        
        **kwargs : dict, optional
            dictionary of optional parameters to be passed to the simulation code, by default {}
                including:
                    num_points : int, optional
                        number of points in the trap distribution, by default 20
                    ignore_error_code : bool, optional
                        if True, ignore the error code returned by SIMsalabim, by default True
                    take_log : bool, optional
                        if True, take the log of the absolute value of the current density, by default False

        Returns
        -------
        np.array
            Array of containing the simulation results (Current Density)
        """  

        take_log = kwargs.get('take_log',False) # if True, take the log of the current density
        num_points = kwargs.get('num_points',20) # number of points in the trap distribution
        ignore_error_code = kwargs.get('ignore_error_code',True) # if True, ignore the error code returned by SIMsalabim

        
        if dev_par_fname=='': # if no dev_par_fname is provided, use the default one
            dev_par_fname = ''
            ParFileDic = ReadParameterFile(os.path.join(self.path2simu,'device_parameters.txt'))
        else:
            ParFileDic = ReadParameterFile(os.path.join(self.path2simu,dev_par_fname))
            fixed_str = dev_par_fname + ' ' + fixed_str # add the dev_par_fname at the beginning of the fixed_str

        fixed_params = {} # dict of structure: {X_dimension:ordered list of unique params}

        # Get the unique values of the experimental parameters
        X_dimx = [XX for XX in X_dimensions if XX!='Vext'] # exclude Vext from the list of experimental parameters
        idimX = [ii for ii,XX in enumerate(X_dimensions) if XX!='Vext'] # get the indices of the experimental parameters without Vext
        
        LU = 1 # need to initialize LU
        for idim,xdim in zip(idimX,X_dimx):
            uniques1,iau = np.unique(X[:,idim],return_index=True) # make sure we don't drop the index of the first occurence of a unique value
            iaus = np.argsort(iau) # sort the indices of the first occurence of a unique value
            uniques = uniques1[iaus] # sort the unique values
            fixed_params[X_dimensions[idim]] = uniques  
            LU = len(uniques) # this should be the same for all          
        
        str_lst,labels,JV_files,Var_files,path_lst,code_name_lst,scPars_files,val,nam,V,JV_files_all = [],[],[],[],[],[],[],[],[],[],[]

        for param in params:
            val.append([param.val]) # param vals in order of param
            nam.append(param.name) # names in order of param

        # check if 'Eu' is in the list of parameters
        if Trap_dist is not None:
            if 'Eu' in nam or 'Eu' in fixed_params.keys():
                if 'Eu' in nam:
                    idx = nam.index('Eu')
                    Eu = val[idx][0]
                elif 'Eu' in fixed_params.keys():
                    Eu = fixed_params['Eu'][0]
                else:
                    raise ValueError('Eu is not in the list of parameters, please add it to the list of parameters')
                    
                if Trap_dist == 'double_urbach_midgap':
                    if 'frac_mid' in nam or 'frac_mid' in fixed_params.keys():
                        if 'frac_mid' in nam:
                            idx = nam.index('frac_mid')
                            frac_mid = val[idx][0]
                        elif 'frac_mid' in fixed_params.keys():
                            frac_mid = fixed_params['frac_mid'][0]
                        else:
                            raise ValueError('frac_mid is not in the list of parameters, please add it to the list of parameters')
                        if frac_mid < 0 or frac_mid > 1:
                            raise ValueError('frac_mid must be between 0 and 1')
                    else:
                        raise ValueError('frac_mid is not in the list of parameters, please add it to the list of parameters')
            else:
                raise ValueError('Eu is not in the list of parameters, please add it to the list of parameters')
                
            if num_points <= 0:
                raise ValueError('num_points must be greater than 0')
            
            num_points = int(num_points) # make sure num_points is an integer

            # create the trap distribution file for SIMsalabim
            CB = ParFileDic['CB'] # conduction band default value
            VB = ParFileDic['VB'] # valence band default value
            if 'CB' in fixed_params.keys(): # if CB is in the list of fixed parameters, use the value of CB
                CB = fixed_params['CB'][0]
            elif 'CB' in nam: # if CB is in the list of parameters, use the value of CB
                idx = nam.index('CB')
                CB = val[idx][0]
            if 'VB' in fixed_params.keys(): # if VB is in the list of fixed parameters, use the value of VB
                VB = fixed_params['VB'][0]
            elif 'VB' in nam: # if VB is in the list of parameters, use the value of VB
                idx = nam.index('VB')
                VB = val[idx][0]

            # make sure CB and VB are floats
            CB = float(CB)
            VB = float(VB)

            if Trap_dist == 'conduction_urbach':
                E,Ntraps = conduction_urbach(CB,VB,Eu,num_points)
            elif Trap_dist == 'valence_urbach':
                E,Ntraps = valence_urbach(CB,VB,Eu,num_points)
            elif Trap_dist == 'double_urbach_midgap':
                E,Ntraps = double_urbach_midgap(CB,VB,Eu,frac_mid,num_points)
            elif Trap_dist == 'double_urbach':
                E,Ntraps = double_urbach(CB,VB,Eu,num_points)
            else:
                raise ValueError('Trap_dist must be ''conduction_urbach'', ''valence_urbach'', ''double_urbach_midgap'' or ''double_urbach''')

            # create trap file
            df_trap = pd.DataFrame({'E':E,'Ntraps':Ntraps})
            filename = f'urbach_{Eu*1e3:.3e}_mV.txt'
            # set numerb of digits to 4
            df_trap.to_csv(os.path.join(self.path2simu,filename),sep='\t',index=False,header=True,float_format='%.4e')

            # append the trap file to fixed_str
            fixed_str = fixed_str + ' -BulkTrapFile ' + filename
        else:
            pass # no trap distribution file
        
        
        Y = []
        Xs = []

        
        exptl_param_names = [k for k,v in fixed_params.items()]
        nam = nam + exptl_param_names
        for ii in range(LU):
            exptl_params = [v[ii] for k,v in fixed_params.items()]
            
            i = [vv[0] for vv in val] + exptl_params
            

            str_line = fixed_str + ' '
            lab = ''
            JV_name = 'JV'
            Var_name = 'none'
            short_dic = {}
            scPars_name = 'scPars'
            dum = []

            # correct the parameters for the relative energy level positions
            CorrectedParams = self.Check_fit_params(i,nam,ParFileDic)
            
            # create the string to be passed to SIMsalabim
            for j,name in zip(i,nam):
                
                if name in CorrectedParams.keys():
                    dumval = CorrectedParams[name]
                else:
                    dumval = j

                if name in ParFileDic.keys(): # don't pass the parameters that are in the ParFileDic to SIMsalabim, but keeo them for the labels
                    str_line = str_line +'-'+name+' {:.3e} '.format(dumval)
                elif name in 'Cions':
                    str_line = str_line +'-CNI {:.3e} -CPI {:.3e} '.format(dumval,dumval)


                JV_name = JV_name +'_'+name +'_{:.3e}'.format(dumval)
                scPars_name = scPars_name +'_'+ name +'_{:.3e}'.format(dumval)
                if name in X_dimensions:
                    if name != 'Vext':
                        dum.append(j)
                        
            Xs.append(dum)
            

            if self.SafeModeParallel:
                JV_name = 'JV_' + str(uuid.uuid4())
                str2run =  str_line+ '-JV_file '+JV_name+ '.dat -Var_file none'
                str_lst.append(str2run)
                JV_files.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                JV_files_all.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                Var_files.append(os.path.join(self.path2simu , str(Var_name+ '.dat')))
                scPars_files.append(os.path.join(self.path2simu , str(scPars_name+ '.dat')))
                code_name_lst.append('simss')
                path_lst.append(self.path2simu)
                labels.append(lab)
            else:
                str2run =  str_line+ '-JV_file '+JV_name+ '.dat -Var_file none'
                str_lst.append(str2run)
                JV_files.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                JV_files_all.append(os.path.join(self.path2simu , str(JV_name+ '.dat')))
                Var_files.append(os.path.join(self.path2simu , str(Var_name+ '.dat')))
                scPars_files.append(os.path.join(self.path2simu , str(scPars_name+ '.dat')))
                code_name_lst.append('simss')
                path_lst.append(self.path2simu)
                labels.append(lab)
        
        Simulation_Input = str_lst,JV_files,Var_files,scPars_files,code_name_lst,path_lst,labels

        RunSimulation(Simulation_Input,max_jobs=max_jobs,do_multiprocessing=True,ignore_error_code=ignore_error_code)

        idx = 0
        for JV_name,fix in zip(JV_files_all,Xs):
            volt = []
            for i in X:
                if (i[1:] == np.asarray(fix)).all():
                    volt.append(i[0])

            # Read JV file output from SIMsalabim
            try:
                df = pd.read_csv(JV_name,delim_whitespace=True)
            except:
                print(str_lst[idx])
                raise Exception('Could not read JV file: ',JV_name)
            
            # Do interpolation in case SIMsalabim did not return the same number of points
            try: # try sline interpolation first and if it fails use simple linear interpolation
                tck = interpolate.splrep(df['Vext'], df['Jext'], s=0)
                Ysin = interpolate.splev(volt, tck, der=0,ext = 0)
            except Exception as e:
                # try linear interpolation
                try:  
                    f = interpolate.interp1d(df['Vext'], df['Jext'],kind='linear',fill_value='extrapolate')
                    Ysin = f(volt)
                except: # if linear interpolation fails, use the minimum value of the JV curve
                    if min(volt)- 0.025 < min(df['Vext']): # need this as a safety to make sure we output something
                        # add a point at the beginning of the JV curve
                        df = df.append({'Vext':min(volt),'Jext':df['Jext'].iloc[0]},ignore_index=True)
                        df = df.sort_values(by=['Vext'])
                    f = interpolate.interp1d(df['Vext'], df['Jext'],kind='linear',fill_value='extrapolate')
                    Ysin = f(volt)

            if take_log:
                Ysin = np.log10(abs(Ysin))

            if len(Y) == 0:
                Y = Ysin
            else:
                Y = np.append(Y,Ysin)
            
            # Delete the JV file
            os.remove(JV_name)
            idx += 1


        return Y


    def Check_fit_params(self,vals,names,ParFileDic):
        """Correct the energy level values from relative positions to absolute positions that are used in SIMsalabim.\\
            Perform the same check as in SIMsalabim to make sure that the energy levels are in the corrected properly.
            
        Parameters
        ----------
        nam : str
            name of the parameter to be checked
        val : float
            value of the parameter to be checked
        names : list
            list of the names of the fitting parameters
        ParFileDic : dict
            dictionary of the parameters in the parameter file
        
        Returns
        -------
        ParStrDic : dict
            dictionary of the parameters that are to be changed in the parameter file or string
        """        
        
        # Energy level names
        nrj_param = ['CB','VB','CB_LTL','VB_LTL','CB_RTL','VB_RTL','W_L','W_R','ETrapSingle']
        

        ParStrDic = {}
        CorrectedDic = {}
        for i,j in zip(names,vals):
            ParStrDic[i] = float(j)
        
        
        # loop through the ParStrDic and check if the value is within the range of the energy levels
        if 'CB' not in ParStrDic.keys():
            CB = float(ParFileDic['CB'])
        else:
            CB = float(ParStrDic['CB'])
            CorrectedDic['CB'] = CB
        
        if 'VB' not in ParStrDic.keys():
            VB = float(ParFileDic['VB'])
        else:
            VB = float(ParStrDic['VB'])
            CorrectedDic['VB'] = VB

        if 'CB_LTL' not in ParStrDic.keys():
            CB_LTL = float(ParFileDic['CB_LTL'])
        else:
            CB_LTL = CB + float(ParStrDic['CB_LTL'])
            CorrectedDic['CB_LTL'] = CB_LTL
        
        if 'VB_LTL' not in ParStrDic.keys():
            VB_LTL = float(ParFileDic['VB_LTL'])
        else:
            VB_LTL = VB + float(ParStrDic['VB_LTL'])
            CorrectedDic['VB_LTL'] = VB_LTL
        
        if 'CB_RTL' not in ParStrDic.keys():
            CB_RTL = float(ParFileDic['CB_RTL'])
        else:
            CB_RTL = CB + float(ParStrDic['CB_RTL'])
            CorrectedDic['CB_RTL'] = CB_RTL

        
        if 'VB_RTL' not in ParStrDic.keys():
            VB_RTL = float(ParFileDic['VB_RTL'])
        else:
            VB_RTL = VB + float(ParStrDic['VB_RTL'])
            CorrectedDic['VB_RTL'] = VB_RTL
        
        if 'W_L' not in ParStrDic.keys():
            W_L = float(ParFileDic['W_L'])
        else:
            if float(ChosePar('L_LTL', ParStrDic, ParFileDic)) > 0:
                W_L = CB_LTL + float(ParStrDic['W_L'])
            else:
                W_L = CB + float(ParStrDic['W_L'])
            CorrectedDic['W_L'] = W_L
        
        if 'W_R' not in ParStrDic.keys():
            W_R = float(ParFileDic['W_R'])
        else:
            if float(ChosePar('L_RTL', ParStrDic, ParFileDic)) > 0:
                W_R = VB_RTL + float(ParStrDic['W_R'])
            else:
                W_R = VB + float(ParStrDic['W_R'])
            CorrectedDic['W_R'] = W_R

        if 'ETrapSingle' not in ParStrDic.keys():
            ETrapSingle = float(ParFileDic['ETrapSingle'])
        else:
            if int(ParFileDic['Tr_type_B' ]) == -1:
                if float(ChosePar('L_LTL', ParStrDic, ParFileDic)) > 0 :
                    ETrapSingle = max(CB,CB_LTL) + float(ParStrDic['ETrapSingle'])
                else:
                    ETrapSingle = CB + float(ParStrDic['ETrapSingle'])
            elif int(ParFileDic['Tr_type_B' ]) == 1:
                if float(ChosePar('L_RTL', ParStrDic, ParFileDic)) > 0 :
                    ETrapSingle = min(VB,VB_RTL) - float(ParStrDic['ETrapSingle'])
                else:
                    ETrapSingle = VB - float(ParStrDic['ETrapSingle'])
            CorrectedDic['ETrapSingle'] = ETrapSingle

        # Make the same checks as in SIMsalabim
        if CB >= VB:
            raise Exception('CB should be smaller than VB.')
        
        if float(ChosePar('L_LTL', ParStrDic, ParFileDic)) > 0 :
            if W_L < CB_LTL:
                raise Exception('W_L cannot be smaller than CB_LTL.')
            if W_L > VB_LTL:
                raise Exception('W_L cannot be larger than VB_LTL.')
            if CB_LTL >= VB_LTL:
                raise Exception('CB_LTL should be smaller than VB_LTL.')

        else:
            if W_L < CB:
                raise Exception('W_L cannot be smaller than CB.')
            if W_L > VB:
                raise Exception('W_L cannot be larger than VB.')
        
        if float(ChosePar('L_RTL', ParStrDic, ParFileDic)) > 0 :
            if W_R < CB_RTL:
                raise Exception('W_R cannot be smaller than CB_RTL.')
            if W_R > VB_RTL:
                raise Exception('W_R cannot be larger than VB_RTL.')
            if CB_RTL >= VB_RTL:
                raise Exception('CB_RTL should be smaller than VB_RTL.')
        else:
            if W_R < CB:
                raise Exception('W_R cannot be smaller than CB.')
            if W_R > VB:
                raise Exception('W_R cannot be larger than VB.')

        
        # check if we have traps
        if float(ChosePar('Bulk_tr',ParStrDic,ParFileDic)) < 0 or float(ChosePar('St_L',ParStrDic,ParFileDic)) < 0 or float(ChosePar('St_R',ParStrDic,ParFileDic)) < 0 or float(ChosePar('GB_tr',ParStrDic,ParFileDic)) < 0:
            raise Exception('Negative trap density not allowed.')
        
        if float(ChosePar('St_L',ParStrDic,ParFileDic)) > 0 and float(ChosePar('L_LTL',ParStrDic,ParFileDic)) == 0:
            raise Exception('You cannot have interface traps (St_L>0) without a TL (L_LTL=0).')
        
        if float(ChosePar('St_R',ParStrDic,ParFileDic)) > 0 and float(ChosePar('L_RTL',ParStrDic,ParFileDic)) == 0:
            raise Exception('You cannot have interface traps (St_R>0) without a TL (L_RTL=0).')
        
        if int(ChosePar('num_GBs',ParStrDic,ParFileDic)) < 0:
            raise Exception('The number of grain boundaries (num_GBs) cannot be negative.')

        if int(ChosePar('num_GBs',ParStrDic,ParFileDic)) > 0 and float(ChosePar('GB_tr',ParStrDic,ParFileDic)) == 0:
            raise Exception('Trap density at grain boundaries (GB_tr) must be > 0 if num_GBs>0.')

        if float(ChosePar('Bulk_tr',ParStrDic,ParFileDic)) > 0 or float(ChosePar('St_L',ParStrDic,ParFileDic)) > 0 or float(ChosePar('St_R',ParStrDic,ParFileDic)) > 0 or float(ChosePar('GB_tr',ParStrDic,ParFileDic)) * int(ChosePar('num_GBs',ParStrDic,ParFileDic)) > 0:

            minEtrap = CB
            maxEtrap = VB
            if int(ChosePar('TLsTrap',ParStrDic,ParFileDic)) == 1 and float(ChosePar('L_LTL', ParStrDic, ParFileDic)) > 0:
                minEtrap = max(minEtrap,CB_LTL)
                maxEtrap = min(maxEtrap,VB_LTL)
            
            if int(ChosePar('TLsTrap',ParStrDic,ParFileDic)) == 1 and float(ChosePar('L_RTL', ParStrDic, ParFileDic)) > 0:
                minEtrap = max(minEtrap,CB_RTL)
                maxEtrap = min(maxEtrap,VB_RTL)
            
            if float(ChosePar('L_LTL', ParStrDic, ParFileDic)) > 0 and float(ChosePar('St_L', ParStrDic, ParFileDic)) > 0:
                minEtrap = max(minEtrap,CB_LTL)
                maxEtrap = min(maxEtrap,VB_LTL)
            
            if float(ChosePar('L_RTL', ParStrDic, ParFileDic)) > 0 and float(ChosePar('St_R', ParStrDic, ParFileDic)) > 0:
                minEtrap = max(minEtrap,CB_RTL)
                maxEtrap = min(maxEtrap,VB_RTL)

            if ETrapSingle >= maxEtrap or ETrapSingle <= minEtrap:
                raise Exception('The value of the trap energy is outside the allowed range of the energy levels. It needs to be between ' + str(minEtrap) + ' and ' + str(maxEtrap) + ' eV but is ' + str(ETrapSingle) + ' eV.')

        return CorrectedDic

        
                
            


                    
                        


                    

