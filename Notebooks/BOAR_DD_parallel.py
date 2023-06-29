
from copy import deepcopy
from boar import *
from sklearn.preprocessing import minmax_scale

import ray


def get_data_transpose(filename):
    # return panda from transposed txt file with first column as header
    with open(filename) as file:
        lines = [x.replace('\n', '').split('\t') for x in file]
    x=np.array(lines)
    names = x[:,0]
    x = np.delete(x, (0), axis=1)
    x = x.astype(np.float)
    df = pd.DataFrame(x.T,columns = names)
    return df

def filter_data(df):
    # filter data
    Voc = get_Voc(df['V'],df['J'])
    Jsc = get_Jsc(df['V'],df['J'])
    # df1 = df[df['V'] >= 0 ]
    df2 = df[df['J'] <= 1*abs(Jsc)]

    return df2

def get_JVs(ODs,t,path2data,Vlim = [],plot=False):
    # get data from all files
    dat = {}
    Xs,ys,weights = [],[],[]
    viridis = plt.get_cmap('viridis', len(ODs))
    for idx,OD in enumerate(ODs):
        if OD != 'dark':
            filename = '{:.2f}OD_IV.dat'.format(OD)
            if os.path.exists(os.path.join(path2data,filename)):
                JV = get_data_transpose(os.path.join(path2data,filename))
                X = np.asarray(JV['voltage(current/time)'])
                y = JV[t]
                # convert y to A/m2
                y = y*10

                if Vlim != []:
                    dumX,dumy = [],[]
                    for v,j in zip(X,y):
                        if v >= Vlim[0] and v <= Vlim[1]:
                            dumX.append(v)
                            dumy.append(j)
                
                    X = np.asarray(dumX)
                    y = np.asarray(dumy)
                X = np.column_stack((X,10**-OD*np.ones(len(X)))) # add Gfrac as second column

                if plot:
                    plt.subplot(1, 2, 1)
                    plt.plot(X[:,0],y,'-',label='Gfrac = '+str(10**-OD),color=viridis(idx))                   
                    plt.ylim([-240,10])
                    plt.xlabel('V [V]')
                    plt.ylabel('Current density [A/m$^2$]')

                if 10**-OD > 0:
                    power = -X[:,0]*y
                    power = minmax_scale(np.asarray(power), feature_range=(1, 100))
                    weight = list(power)
                else:
                    power = np.ones(len(y))

                if plot:
                    plt.subplot(1, 2, 2)
                    plt.plot(X[:,0],power,'-',label='Gfrac = '+str(10**-OD),color=viridis(idx))
                    plt.xlabel('V [V]')
                    plt.ylabel('Weights')
           

                if Xs == []:
                    Xs = X
                    ys = y
                    weights = weight
                else:
                    Xs = np.vstack((Xs,X))
                    ys = np.hstack((ys,y))
                    weights = np.hstack((weights,weight))
   
            else :
                warnings.warn('File {} does not exist'.format(filename))
        else:
            filename = 'dark_IV.dat'
            if os.path.exists(os.path.join(path2data,filename)):
                JV = get_data_transpose(os.path.join(path2data,filename))
                X = np.asarray(JV['voltage(current/time)'])
                y = JV[t]
                # convert y to A/m2
                y = y * 10
                if Vlim != []:
                    dumX,dumy = [],[]
                    for v,j in zip(X,y):
                        if v >= Vlim[0] and v <= Vlim[1]:
                            dumX.append(v)
                            dumy.append(j)
                    X = np.asarray(dumX)
                    y = np.asarray(dumy)
                X = np.column_stack((X,0*np.ones(len(X)))) # add Gfrac as second column

                if plot:
                    plt.subplot(1, 2, 1)
                    plt.plot(X[:,0],y,'-',label='Gfrac = 0',color=viridis(idx))                   
                    plt.xlabel('V [V]')
                    plt.ylabel('Current density [A m$^{-2}$]')

                if Xs == []:
                    Xs = X
                    ys = y
                    weights = 1*np.ones(len(y))
                else:
                    Xs = np.vstack((Xs,X))
                    ys = np.hstack((ys,y))
                    weights = np.hstack((weights,1*np.ones(len(y))))
            else :
                warnings.warn('File {} does not exist'.format(filename))


    Xs = np.asarray(Xs)
    ys = np.asarray(ys)
    weights = np.asarray(weights)
    return Xs,ys,weights

@ray.remote
def run_BOAR_DD_parallel(t,ODs,warmstart):


    path2simu = '/home/vlc/Desktop/Paul/SIMsalabimv445_paul/SimSS/'
    path2JV = '/home/vlc/Desktop/Paul/Paul_exp_data/'
    res_dir = '/home/vlc/Desktop/Paul/Fit_Results/'
    # SaveOldXY2file = os.path.join('/home/vlc/Desktop/Paul/Fit_Results','XY_old.json')
    # Path2OldXY = os.path.join('/home/vlc/Desktop/Paul/Fit_Results','XY_old.json')




    # define Fitparameters
    Start_values = {'kdirect':5e-18,'mun_0':2e-8,'mup_0':8e-8,'Nc':5e26,'Gehp':1.28e28,'Bulk_tr':1e20,'Gehp':1.28e28}
    params = []
    Bulk_tr = Fitparam(name = 'Bulk_tr', val = Start_values['Bulk_tr'] , relRange = 0.2, lims=[1e17,5e19],range_type='log',optim_type='log',lim_type='absolute',display_name='N$_{Tr}$ [m$^{-3}$]') #real 6.4e18 
    params.append(Bulk_tr)
    # Bulk_tr.startVal = 2e20

    # kdirect = Fitparam(name = 'kdirect', val = Start_values['kdirect'] , relRange = 0.2, lims=[1e-19,1e-16],range_type='log',optim_type='log',lim_type='absolute',display_name='k$_{2}$ [m$^{3}$ s$^{-1}$]')
    # params.append(kdirect)
    #kdirect.startVal = 5e-18

    Lang_pre = Fitparam(name = 'Lang_pre', val = 1e-1 , relRange = 0.2, lims=[1e-3,1],range_type='log',optim_type='log',lim_type='absolute',display_name='$\gamma_{pre}$')
    params.append(Lang_pre)

    mun_0 = Fitparam(name = 'mun_0', val = Start_values['mun_0'] , relRange = 0.2, lims=[5e-10,1e-6],range_type='log',optim_type='log',lim_type='absolute',display_name='$\mu_n$ [m$^{2}$ V$^{-1}$ s$^{-1}$]')
    params.append(mun_0)
    #mun_0.startVal = 1e-8

    mup_0 = Fitparam(name = 'mup_0', val = Start_values['mup_0'] , relRange = 1.5, lims=[5e-10,1e-6],range_type='log',optim_type='log',display_name='$\mu_p$ [m$^{2}$ V$^{-1}$ s$^{-1}$]')
    params.append(mup_0)
    #mup_0.startVal = 1e-8

    Nc = Fitparam(name = 'Nc', val = 6e26 , relRange = 0, lims=[1e26,1e27],range_type='log',optim_type='log',lim_type='absolute',display_name='N$_{c}$ [m$^{-3}]$')
    params.append(Nc)
    #Nc.startVal = 5e26

    St_L = Fitparam(name = 'St_L', val = 1e11 , relRange = 0.2, lims=[1e10,1e15],range_type='log',optim_type='log',lim_type='absolute',display_name='S$_{Tr}^{ETL}$ [m$^{-2}$]')
    params.append(St_L)

    # mob_LTL = Fitparam(name = 'mob_LTL', val = 1e-6 , relRange = 0.2, lims=[1e-9,1e-5],range_type='log',optim_type='log',lim_type='absolute')
    # params.append(mob_LTL)

    Rseries = Fitparam(name = 'Rseries', val = 1e-4, relRange = 1, lims=[1e-5,1e-1],range_type='log',optim_type='log',lim_type='absolute')
    params.append(Rseries)
    # Rseries.startVal = 2e-3

    Rshunt = Fitparam(name = 'Rshunt', val = 3e2 , relRange = 1, lims=[1e0,1e3],range_type='log',optim_type='log',lim_type='absolute')
    params.append(Rshunt)

    W_R = Fitparam(name = 'W_R', val = 0 , relRange = 1, lims=[-0.2,0],range_type='linear',optim_type='linear',lim_type='absolute',display_name='W$_R$ [eV]')
    params.append(W_R)

    W_L = Fitparam(name = 'W_L', val = 0 , relRange = 0, lims=[0,0.2],range_type='linear',optim_type='linear',lim_type='absolute',display_name='W$_L$ [eV]')
    params.append(W_L)

    CB_LTL = Fitparam(name = 'CB_LTL', val = 0 , relRange = 1, lims=[0,0.3],range_type='linear',optim_type='linear',lim_type='absolute',display_name='CB$_{ETL}$ [eV]')
    params.append(CB_LTL)

    Gehp = Fitparam(name = 'Gehp', val = Start_values['Gehp'] , relRange = 0.2, lims=[1e28,2e28],range_type='log',optim_type='linear',lim_type='absolute',display_name='G$_{ehp}$ [m$^{-3}$ s$^{-1}$]')
    params.append(Gehp)

    kf = Fitparam(name = 'kf', val = 1e8 , relRange = 0.2, lims=[1e6,1e8],range_type='log',optim_type='log',lim_type='absolute',display_name='k$_f$ [s$^{-1}$]')
    params.append(kf)

    # P0 = Fitparam(name = 'P0', val = 0.5 , relRange = 0.2, lims=[0,0.99],range_type='linear',optim_type='linear',lim_type='absolute',display_name='P$_0$ [-]')
    # params.append(P0)

    params_true = copy.deepcopy(params)




    # Define figures-of-merit (FOMs)
    FOMs = []
    FOMs.append(FOMparam(Theta_B, name = 'Theta_B', display_name='$\\theta_B$', optim_type = 'log'))
    FOMs.append(FOMparam(delta_B, name = 'delta_B', display_name='$\delta_B$', optim_type = 'log'))
    FOMs.append(FOMparam(Theta_T, name = 'Theta_T', display_name='$\\theta_T$', optim_type = 'log'))
    FOMs.append(FOMparam(delta_T, name = 'delta_T', display_name='$\delta_T$', optim_type = 'log'))


    FOM_names = [fom.display_name for fom in FOMs] # names of the FOMs


    # Get dark JV and fit nonIdeal diode model
    # dio = Non_Ideal_Diode_agent() # create an instance of the nonIdeal diode model
    # X,y,w = get_JVs(['dark'],t,path2JV)
    # res = dio.FitNonIdealDiode(X[:,0],y,T=300,JV_type='dark',take_log=True,bounds=([1e-20, 0.8, 1e-8, 1e-3], [1e-3, 3, 1e2, 1e8]),p_start={'J0':1e-6})
    # Rseries =res['Rs']
    # Rshunt = res['Rsh']
    
    # # Update the device_parameters.txt file with the new thickness
    # ParFileDic = ReadParameterFile(os.path.join(path2simu, 'device_parameters.txt')) # read the parameters from the file
    # ParFileDic['Rseries'] = Rseries
    # ParFileDic['Rshunt'] = Rshunt
    # fixed_str = '-Rseries {:.5e} -Rshunt {:.5e} '.format(Rseries,Rshunt) # string to be added to the command line
    fixed_str = ''
    # Get experimental data light JV
    X,y,weights = get_JVs(ODs,t,path2JV,Vlim=[-0.2,1])
    X_dimensions = ['Vext','Gfrac']

    # Define weighting for the different JV curves
    use_weighting = True
    if use_weighting:
        weight = weights
    else:
        weight = 1

    # initialize the simulation agent
    dda = Drift_diffusion_agent(path2simu=path2simu) 

    NewName = ''
    target = {'model':partial(dda.DriftDiffusion_relative,X_dimensions=X_dimensions,max_jobs=3,fixed_str=fixed_str,dev_par_fname=NewName),'data':{'X':X,'y':y,
                'X_dimensions':['Vext','Gfrac'],'X_units':['V','sun'],'y_dimension':'Current density','y_unit':r'$A m^{-2}$'}
                ,'params':copy.deepcopy(params), 'weight':weight}

    # Define optimizer
    mo = MultiObjectiveOptimizer(res_dir=res_dir,params=params,targets=[target],SaveOldXY2file=os.path.join('/home/vlc/Desktop/Paul/Fit_Results','XY_old.json'),Path2OldXY=os.path.join('/home/vlc/Desktop/Paul/Fit_Results','XY_old.json'))


    mo.warmstart = warmstart # 'recall' data from Path2OldXY file

    # Define the number of iterations for the optimization
    n_jobs = 4
    n_jobs_init = 20
    n_yscale= 20
    n_initial_points = 40
    if warmstart == 'collect_init':
        n_BO = 0
        n_BO_warmstart = 0
    else:
        n_BO = 120
        n_BO_warmstart = 120
 

    kwargs = {'check_improvement':'relax','max_loop_no_improvement':25,'xtol':1e-3,'ftol':1e-3}
    kwargs_posterior = {'Nres':3,'gaussfilt':1,'logscale':False,'vmin':1e-100,'zoom':0,'min_prob':1e-40,'clear_axis':True,'show_points':True,'savefig':True,'figname':'param_posterior_' + t ,'show_fig':False,'figsize':(14,14)}
    kwargs_plot_obj = {'zscale':'linear','show_fig':False}

    r = mo.optimize_sko_parallel(n_jobs=n_jobs,n_yscale=n_yscale, n_BO=n_BO, n_initial_points = n_initial_points,n_BO_warmstart=n_BO_warmstart,n_jobs_init=n_jobs_init,kwargs=kwargs,verbose=False,loss='linear',threshold=1000,base_estimator = 'GP',show_objective_func=False,show_posterior=True,kwargs_posterior = kwargs_posterior,kwargs_plot_obj=kwargs_plot_obj)
    # pf.append(deepcopy(target['params'])) # collects optimized fitparameters
    rrr = r['r'] # the results dict of the last optimizer.tell()

    best_params = copy.deepcopy(mo.params) # get the best parameters


    # Plot and save the results
    kwargs_JV_plot = {'savefig':True,'figname':'fits_JV_' + t,'figdir':res_dir,'show_fig':False,'figsize':(10,8)}
    dda.Compare_JVs_exp(target,best_params,FOMs=[],verbose=True,ylim=[],fixed_str=NewName,DD_func=partial(dda.DriftDiffusion_relative,X_dimensions=X_dimensions,max_jobs=4,fixed_str=fixed_str,dev_par_fname=NewName),kwargs=kwargs_JV_plot) #-100,10

    Jfit = dda.DriftDiffusion_relative(X,best_params,X_dimensions=X_dimensions,max_jobs=4,fixed_str=fixed_str,dev_par_fname=NewName)

    data = np.concatenate((X, y.reshape(len(y),1), Jfit.reshape(len(Jfit),1)), axis=1)

    # prepare the data for saving
    param_dict = dda.get_param_dict(best_params) # get fitparameters (and fixed ones)
    # param_dict.append({'name':'Rseries','relRange':0,'val':Rseries,'std_l':res['Rs_err'],'std_h':res['Rs_err']})
    # param_dict.append({'name':'Rshunt','relRange':0,'val':Rshunt,'std_l':res['Rsh_err'],'std_h':res['Rsh_err']})
    pout = [[f'{v:.3E}' if isinstance(v,float) else v for _,v in pp.items()] for pp in param_dict]
    # print(mo.targets[0]['params'])


    # produce output excel file with data, fitparameters and FOMs
    fn_xlsx = 'fits_' + t + '.xlsx'
    namecols = X_dimensions + ['Jexp','Jfit']
    # delete old file if it exists
    if os.path.exists(os.path.join(res_dir,fn_xlsx)):
        os.remove(os.path.join(res_dir,fn_xlsx))

    with pd.ExcelWriter(os.path.join(res_dir,fn_xlsx), mode='w') as writer:
        df = pd.DataFrame(data,columns=namecols)
        df.to_excel(writer, sheet_name = 'data') 
        df = pd.DataFrame(pout,columns=[k for k in param_dict[0].keys()])
        df.to_excel(writer, sheet_name = 'params') 



def main():
    ray.init(num_cpus=10)
    # Path to data and SIMsalabim
    path2simu = '/home/vlc/Desktop/Paul/SIMsalabimv445_paul/SimSS/'
    path2data = '/home/vlc/Desktop/Paul/Paul_exp_data/'
    res_dir = '/home/vlc/Desktop/Paul/Fit_Results/'
    cur_dir = os.getcwd()

    # Prepare data
    ODs = np.arange(0,4,0.1)
    dum = []
    for OD in ODs: # remove ODs that do not exist
        filename = '{:.2f}OD_IV.dat'.format(OD)
        if os.path.exists(os.path.join(path2data,filename)):
            dum.append(OD)
    ODs = dum[::-1] # reverse order
    filename = '{:.2f}OD_IV.dat'.format(0)
    JV = get_data_transpose(os.path.join(path2data,filename))
    times = [float(x) for x in JV.keys() if x != 'voltage(current/time)']
    times_str = [x for x in JV.keys() if x != 'voltage(current/time)']
    ODs = ['dark'] + ODs
    # X,y,w = get_JVs([1,0.3,0],times_str[0],path2data,Vlim = [-0.2,1],plot=False)

    # Fit data
    ODs2fit =[1,0.3,0]
    log_spaced = np.geomspace(times[1],times[-1],num=30)
    # find the closest time to the log spaced times and add 0 at the beginning
    # times2fit = [times_str[0]] + [times_str[np.argmin(np.abs(np.asarray(times)-x))] for x in log_spaced]
    # times2fit = list(dict.fromkeys(times2fit)) # drop duplicates
    times2fit = times_str
    times =[float(x) for x in times2fit] 

    # run BOAR for initial sampling
    rerun_simu = True
    if rerun_simu:
        print('Start initial sampling')
        start_time = time() #time,ODs,path2simu='',path2JV ='',res_dir='temp',warmstart='recall',SaveOldXY2file = '',Path2OldXY='')
        kwargs = {'n_BO':0,'warmstart':'collect_init'}
        # run_BOAR_DD_parallel(times2fit[0],ODs2fit,'collect_init')
        
        results = run_BOAR_DD_parallel.remote(times2fit[0],ODs2fit,'collect_init') 
        ray.get(results)
        # ray.shutdown()
        print('Finished initial sampling in ' + str(time()-start_time) + ' s')

    # # run BOAR on all times
    rerun_simu = True
    if rerun_simu:
        print('Start BOAR')
        start_time = time()
        curr_dir = os.getcwd()
   
        # use ray for parallelization
        # ray.init()
        results = [run_BOAR_DD_parallel.remote(t,ODs2fit,'recall') for t in times2fit]
        ray.get(results)
        ray.shutdown()

        print('Finished BOAR in ' + str(time()-start_time) + ' s')

    
    # Plot JV Exp and Simu data
    fig = plt.figure(100,figsize=(15,10))
    ax = plt.axes()
    # ax = ax.flatten()
    linestyles = ['-','--','-.',':']
    ODs2fit = [0]
    fit = [ODs2fit[0]]
    suns = 10**-np.asarray(ODs2fit)
    viridis = cm.get_cmap('viridis', len(times2fit))
    for i,t in enumerate(times2fit):
        
        fn_xlsx = 'fits_' + t + '.xlsx'
        # check 
        if os.path.exists(os.path.join(res_dir,fn_xlsx)):
            dff = pd.read_excel(os.path.join(res_dir,fn_xlsx),sheet_name='data')
        

            # filter by Gfrac = suns
            for idx,sun in enumerate(suns):
                dffsun = dff[dff.Gfrac==sun]
                if i == 0 and idx == 0:
                    labelexp = 'Exp.'
                    labelsimu = 'Simu.'
                else:
                    labelexp = None
                    labelsimu = None
                ax.plot(dffsun.Vext,dffsun.Jexp/10,label=labelexp,c=viridis(i),markersize=15,fillstyle='none',marker='o',linestyle='none')
                ax.plot(dffsun.Vext,dffsun.Jfit/10,linestyle=linestyles[0],c=viridis(i),label=labelsimu)

            ax.set_xlabel('V [V]')
            ax.set_ylabel('Current density [mA cm$^{-2}$]')
            ax.legend(loc = 'upper left',fontsize=25)
            ax.set_xlim(-0.2,1.05)
            ax.set_ylim(-24,5)


    plt.tight_layout()  
    plt.savefig(os.path.join(res_dir,'JV_all.png'),dpi=300)




if __name__ == '__main__':
    main()