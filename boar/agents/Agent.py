from copy import deepcopy
import itertools,os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from boar.core.funcs import get_unique_X,get_unique_X_and_xaxis_values

class Agent():
    """ A base class providing functionality such as plotting, export, that all agents need

    """
    def __init__(self) -> None:
        pass

    def get_param_dict(self, params):
        """get standard deviations from parameters
        if optimization has been done in log scale, convert them into 
        linear scale which will make the upper and lower limit unsymmetric and will return a tuple

        Parameters
        ----------
        params : list of Fitparam()
            
        Returns:
         paramdict: list of dict with keys 'name'(string),'relRange'(float),'val'(float),'std'(float or tuple of float)
         depending on whether in the param optim_type=='linear' or 'log', respectively
        """

        paramdict = []
        for param in params:
            p = param.val
            pn = param.name
            if param.lims ==[]:
                if param.relRange == 0:
                    rr = 0.001
                else:
                    rr = param.relRange
                lims = [param.val - rr*abs(param.val),param.val + rr*abs(param.val)]
            else:
                lims = param.lims


            if param.relRange!=0:
                s = param.std
                if param.optim_type=='log':# obj func is interpreted as log normal 
                    # s[s>2]=2
                    s1 = 10**(np.log10(p)+s) - p # upper error bars 
                    s2 = -10**(np.log10(p)-s) + p # lower error bars 
                    # check if the error bars are within the limits else set them to the limit
                    if p+s1>lims[1]:
                        s1 = lims[1]
                    if p-s2<lims[0]:
                        s2 = lims[0]

                    s3 = (s2,s1)
                else:
                    s3 = (s,s)
            else:
                s3 = (0,0)
            paramdict.append({'name':pn,'relRange':param.relRange,'val':p,'std_l':s3[0],'std_h':s3[1]})

        return paramdict

    def get_fitval(self,ps,params):
        """Get the values for all fitparam objects in params whoes names are mentioned in ps

        Parameters
        ----------
        ps : ist of string
            list the names of the desired fit parameters
        params : list of Fitparam()
            Fitparam objects

        Returns
        -------
        ist of float
            the values of the respective fit parameters
        """
        out = []
        for s in ps:
            out.append([pp.val for pp in params if pp.name==s][0])
        return out

    def Xy_to_dict(self,X,yexp=[],yfit=[],axis=0,X_dimensions=[]):
        """Convert the X array into a dict containing the measured axis and the (constant) metadata
        This representation is useful for physical modeling (rate equations, drift-diffusion, etc)
        while the X notation is more useful for machine learning.

        Parameters
        ----------
        X : ndarray
           the experimental dimensions
        yexp : ndarray of shape(n,)
            the experimental values, if available
        yfit : ndarray of shape(n,)
            the fitted values, if available
        axis : int, optional
            the column in X containing the varied parameter, by default 0
        X_dimensions : list, optional
            names of the X columns (including the measurement axis), by default []

        Returns
        -------
        dictionary
            key 'x' is independent parameter, oter keys are metadata
        """

        Xm = np.delete(X, axis, 1)  # delete data column of X, leaving only the metadata columns
        uniqs = np.unique(Xm,axis=0)
        X_dimensions = deepcopy(X_dimensions) # dereference
        if len(X_dimensions)==0:
            X_dimensions = ['dim_'+str(n) for n in range(X.shape[1])]
        X_data_dim = X_dimensions.pop(axis)
        outdict= []
        for uniq in uniqs:
            metadata = {}
            for pp,Xd in zip(uniq,X_dimensions):
                metadata.update({Xd:pp})

            irs = [] # initialize and conditions
            for ii,pp in enumerate(uniq):   
                ir = np.where(Xm[:,ii]==pp)[0]# apply an AND condition of unknown nb of items
                irs = irs + list(ir)

            irsu,irsuc = np.unique(np.array(irs),return_counts=True)
            irsux = irsu[irsuc==len(uniq)] # for AND it must appear in all columns

            ydict = {'yexp':[],'yfit':[]}
            if len(yexp)>0:
                ydict.update({'yexp':yexp[irsux]})
            if len(yfit)>0:
                ydict.update({'yfit':yfit[irsux]})
            dum_dict = {'x':X[irsux,axis]}
            dum_dict.update(metadata)
            dum_dict.update(ydict)
            outdict.append(dum_dict)
        return outdict 

    def plot_results(self,outdict,pdisplay,offset=0,X_dimensions=[],y_dimension = ''):
        if len(X_dimensions)==0:
            X_dimensions = ['X axis']
        if len(y_dimension)==0:
            y_dimension = ['Y axis']            
        fig = plt.figure(figsize=(8,8))
        for ii,curve in enumerate(outdict):
            plt.plot(curve['x'],curve['yexp']+ii*offset,linewidth=1, alpha=0.4,c='C'+str(ii))
            plt.plot(curve['x'],curve['yfit']+ii*offset,linewidth=2,c='C'+str(ii),label = f'{curve[pdisplay]:.2e}' if pdisplay in curve.keys() else '')
            plt.plot(curve['x'],curve['yfit']*0+ii*offset,'--',linewidth=1,c='C'+str(ii))
        plt.xlabel(X_dimensions[0])
        plt.ylabel(y_dimension)
        plt.legend(title=pdisplay)
        return fig

    def to_excel(self,fn_xlsx,outdict,params):
        param_dict = self.get_param_dict(params) # get fitparameters (and fixed ones)
        pout = [[f'{v:.3E}' if isinstance(v,float) else v for _,v in pp.items()] for pp in param_dict]

        with pd.ExcelWriter(fn_xlsx, mode='w') as writer:
            for ii,outx in enumerate(outdict):
                data = np.array([outx['x'],outx['yexp'],outx['yfit']]).T
                df = pd.DataFrame(data,columns=['x','yexp','yfit'])
                df.to_excel(writer, sheet_name = 'data_'+str(ii)) 
                mdk = [k for k in outx.keys() if k not in ['x','yexp','yfit']]
                mdv = [v for k,v in outx.items() if k not in ['x','yexp','yfit']]
                df = pd.DataFrame(np.array([mdk,mdv]).T,columns=['par','val'])
                df.to_excel(writer, sheet_name = 'metadata_'+str(ii))
            df = pd.DataFrame(pout,columns=[k for k in param_dict[0].keys()])
            df.to_excel(writer, sheet_name = 'params') 
    
 
    def plot_params(self,paramslist,fpu=[],kwargs=None):
        """_summary_

        Parameters
        ----------
        paramslist : list
            list of FitParam objects
        fpu : list, optional
            list to plot on the x axis, by default []
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
                    size of the figure, by default (8*Np,12)
                figdpi : int, optional
                    dpi of the figure, by default 300
                nrows : int, optional
                    number of rows in the figure, by default 1
                ncols : int, optional
                    number of columns in the figure, by default Np

        Returns
        -------
        tuple
            tuple containing:
                fpu
        """   
        infopack = [] # for output
        pf = [[p for p in pp  if p.relRange!=0] for pp in paramslist] # free params
        Np = len(pf[0]) 
        if len(fpu)==0:
            fpu = range(len(paramslist))

        if kwargs is None:
            kwargs = {}
        savefig = kwargs.get('savefig',True)
        figname = kwargs.get('figname','JV_fit')
        figdir = kwargs.get('figdir','')
        figext = kwargs.get('figext','.png')
        figsize = kwargs.get('figsize',(16,12))
        figdpi = kwargs.get('figdpi',300)   
        xaxis_label = kwargs.get('xaxis_label','# meas')
        figsize = kwargs.get('figsize',(8*Np,12))
        nrows = kwargs.get('nrows',1)
        ncols = kwargs.get('ncols',Np)

        fig,axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        plt.subplots_adjust(wspace=0.4,hspace=0.4,bottom=0.2,left=0.2)
        axs = axs.flatten()
        for ii in range(Np): # iterate over all parameters 
            nam = [r[ii].display_name for r in pf][0]
            p = np.array([r[ii].val for r in pf])
            s = np.array([r[ii].std for r in pf])
            lims = np.array([r[ii].lims for r in pf])
            if pf[0][ii].optim_type=='log':# obj func is interpreted as log normal 
                axs[ii].set_yscale('log') 
                # s[s>2]=2
                
                s1 = 10**(np.log10(p)+s) - p # upper error bars
                s2 = -10**(np.log10(p)-s) + p # lower error bars 

                # check if error bars are outside the limits
                s1[p+s1>lims[:,1]]=lims[p+s1>lims[:,1],1]
                s2[p-s2<lims[:,0]]=lims[p-s2<lims[:,0],0]

                s3 = np.vstack((s2.reshape(1,-1),s1.reshape(1,-1)))
                
            else:
                s3 = s
            s3[np.isinf(s3)]=0 # set error bars to zero if std is inf
            s3[np.isnan(s3)]=0 # set error bars to zero if std is nan
            
            axs[ii].errorbar(fpu, p, yerr=s3) 
            axs[ii].plot(fpu, p, 'o',c='C0')
            axs[ii].set_xlabel(xaxis_label)
            #axs[ii].set_xticks([0,1,2,3]) 
            #axs[ii].set_xticklabels(['a','b','c','d'])
            #axs[ii].set_xscale('log')
            infopack.append([fpu,p,s3])
            axs[ii].set_ylabel(nam)
        plt.tight_layout()
        plt.show()

        if savefig:
            fig.savefig(os.path.join(figdir,figname+figext), dpi=figdpi)

        return (infopack)

    # def get_unique_X(self,X,xaxis,X_dimensions):
    #     """Get the unique values of the independent variable (X) in the dataset

    #     Parameters
    #     ----------
    #     X : ndarray
    #         the experimental dimensions
    #     xaxis : str, optional
    #         the name of the independent variable
    #     X_dimensions : list, optional
    #         names of the X columns
    #     Returns
    #     -------
    #     X_unique : ndarray
    #         the unique values of the independent variable
    #     X_dimensions_uni : list
    #         the names of the columns of X_unique

    #     Raises
    #     ------
    #     ValueError
    #         if xaxis is not in X_dimensions

    #     """
    #     X_unique = deepcopy(X)
    #     idx_x = None
    #     if xaxis in X_dimensions:
    #         idx_x = X_dimensions.index(xaxis)
    #     else:
    #         raise ValueError(xaxis + ' not in X_dimensions, please add it to X_dimensions')

    #     X_unique = np.delete(X_unique,idx_x,axis=1)
    #     X_dimensions_uni = [x for x in X_dimensions if x != xaxis]
    #     # get index of unique values
    #     unique,idxuni = np.unique(X_unique,axis=0,return_index=True)

    #     X_unique = X_unique[np.sort(idxuni),:] # resort X_unique 

    #     return X_unique,X_dimensions_uni

    # def get_unique_X_and_xaxis_values(self,X,xaxis,X_dimensions):
    #     """Get the values of the independent variable (X) in the dataset for each unique value of the other dimensions

    #     Parameters
    #     ----------
    #     X : ndarray
    #         the experimental dimensions
    #     xaxis : str, optional
    #         the name of the independent variable
    #     X_dimensions : list, optional
    #         the names of the columns of X
    #     Returns
    #     -------
    #     xs : list of ndarrays
    #         the values of the independent variable for each unique value of the other dimensions

    #     """
    #     X_unique, X_dimensions_uni = self.get_unique_X(X,xaxis,X_dimensions) # get unique X values and their dimensions
    #     idx_x = int(X_dimensions.index(xaxis))
    #     xs = []
    #     for uni in X_unique:
            
    #         X_dum = deepcopy(X)
    #         # drop the xaxis column
    #         X_dum = np.delete(X_dum,X_dimensions.index(xaxis),axis=1)
    #         # find indexes where the other columns are equal to the unique values
    #         idxs = np.where(np.all(X_dum==uni,axis=1))[0]
    #         # get the values of the xaxis for these indexes
    #         xs.append(X[idxs,idx_x])
            
            

    #     return X_unique, X_dimensions_uni, xs

    def plot_fit_res(self,target,params,xaxis_name,xlim=[],ylim=[],kwargs=None):
        """Compare the targets vs fitting results

        Parameters
        ----------
        target : dict
            Dictionary with the target parameters
        params :list
            list of Fitparam objects with the fitted parameters
        xaxis_name : str
            name of the x axis
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
                    size of the figure, by default (8,8)
                figdpi : int, optional
                    dpi of the figure, by default 300
        """  
        # if kwargs != {}:
        if kwargs is None:
            kwargs = {}
        show_fig = kwargs.get('show_fig',True)
        savefig = kwargs.get('savefig',True)
        figname = kwargs.get('figname','JV_fit')
        figdir = kwargs.get('figdir','')
        figext = kwargs.get('figext','.png')
        figsize = kwargs.get('figsize',(8,8))
        figdpi = kwargs.get('figdpi',300)
        x_scaling = kwargs.get('x_scaling',1)
        y_scaling = kwargs.get('y_scaling',1)
        xscale_type = kwargs.get('xscale_type','linear')
        yscale_type = kwargs.get('yscale_type','linear')
        norm_data = kwargs.get('norm_data',False)
        delog = kwargs.get('delog',False)

        # check if xscale_type and yscale_type are correct
        if xscale_type == 'lin':
            xscale_type = 'linear'
        if yscale_type == 'lin':
            yscale_type = 'linear'


        if 'xaxis_label' in kwargs.keys():
            xaxis_label = kwargs['xaxis_label']
        else:
            xaxis_label = xaxis_name + ' [' + target['data']['X_units'][target['data']['X_dimensions'].index(xaxis_name)] + ']'
        
        if 'yaxis_label' in kwargs.keys():
            yaxis_label = kwargs['yaxis_label']
        else:
            yaxis_label = target['data']['y_dimension'] + ' ' + target['data']['y_unit']

        # get unique X values and their dimensions
        X_unique, X_dimensions_uni, xs = get_unique_X_and_xaxis_values(target['data']['X'],xaxis_name,target['data']['X_dimensions'])

        # get exp values
        ytrue = target['data']['y']

        # run model with fitted parameters
        yfit = target['model'](target['data']['X'],params,X_dimensions=target['data']['X_dimensions'] )

        # plot
        f = plt.figure(figsize=figsize)
        for idx,uni in enumerate(X_unique):
            lab = ''

            for ii, name in enumerate(X_dimensions_uni):
                if uni[ii] > 1e3 or uni[ii] < 1e-2:
                    dum = '{:.2e}'.format(uni[ii])
                else:
                    dum = '{:.3f}'.format(uni[ii])

                lab = lab + name + ' = ' + dum + ' ' + target['data']['X_units'][target['data']['X_dimensions'].index(name)] + ' '
            
            XX = deepcopy(target['data']['X'])
            XX = np.delete(XX,target['data']['X_dimensions'].index(xaxis_name),axis=1)

            pick = XX==uni
            pick2 = []
            for i in pick:
                if i.all() == True:
                    pick2.append(True)
                else:
                    pick2.append(False)

            xp = target['data']['X'][pick2,0]
            yp = ytrue[pick2]
            yf = yfit[pick2]

            if x_scaling != 1: # rescaling the x axis to chnage the units
                xp = xp*x_scaling

            if y_scaling != 1: # rescaling the y axis to chnage the units
                yp = yp*y_scaling
                yf = yf*y_scaling

            if delog:
                yp = 10**yp
                yf = 10**yf

            if norm_data:
                yp = yp/np.max(yp)
                yf = yf/np.max(yf)
            
            plt.plot(xp,yp,'o',c='C'+str(idx),label=lab,alpha=0.5)
            plt.plot(xp,yf,c='C'+str(idx))
            # plt.loglog(abs(xp),abs(yp),'o',c='C'+str(idx),label=lab)
            # plt.loglog(abs(xp),abs(yf),c='C'+str(idx))
            idx += 1

        if xlim!=[]:
            plt.xlim(xlim)
        if ylim!=[]:
            plt.ylim(ylim)
        plt.xlabel(xaxis_label)
        plt.ylabel(yaxis_label)
        plt.xscale(xscale_type)
        plt.yscale(yscale_type)
        plt.legend(loc='best')
        if show_fig:
            plt.show()

        if savefig:
            f.savefig(os.path.join(figdir,figname+figext), dpi=figdpi)
