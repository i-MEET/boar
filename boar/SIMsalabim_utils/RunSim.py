######################################################################
############### Functions to run simulation code #####################
######################################################################
# Author: Vincent M. Le Corre
# Github: https://github.com/VMLC-PV

# Package import
# import subprocess,shutil,os,tqdm,parmap,multiprocessing,random,sys,platform,itertools,shutil,uuid,warnings
import subprocess,shutil,os,parmap,sys,platform,itertools,shutil,uuid
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
# Import SIMsalabim_utils
from boar.SIMsalabim_utils.CompileProg import *



def run_code(name_prog,path2prog,str2run='',show_term_output=False,verbose=False):
    """Run program 'name_prog' in the folder specified by 'path2prog'.

    Parameters
    ----------
    name_prog : str
        name of the program to run.
    
    path2prog : str
        path to the folder containing  the simulation program.
        (for example './zimt' in Linux and 'zimt.exe' in Windows ).

    st2run : str
        String to run for the name_prog.
    
    show_term_output : bool, optional
        If True, show the terminal output of the program, by default False.
    
    verbose : bool, optional
        Verbose?, by default False.
    
    Returns
    -------
    
    """
    
    System = platform.system()                  # Operating system
    is_windows = (System == 'Windows')          # Check if we are on Windows
    path2prog = str(path2prog)                  # Convert to string
    curr_dir = os.getcwd()                      # Get current directory
    os.chdir(path2prog)                         # Change directory to working directory

    if show_term_output == True:
        output_direct = None
    else:
        output_direct = subprocess.DEVNULL
    
    if is_windows:
        cmd_list = name_prog.lower()+'.exe ' + str2run
        if not os.path.isfile(path2prog+'\\'+name_prog.lower()+'.exe'):
            fpc_prog(name_prog,path2prog,show_term_output=False,force_fpc=False,verbose=verbose)
    else : 
        # cmd_list = './'+name_prog+' ' + str2run
        curr_dir = os.getcwd()                      # Get current directory
        os.chdir(path2prog)                         # Change directory to working directory
        cmd_list = './'+name_prog.lower()+' ' + str2run
        
        if not os.path.isfile('./'+name_prog.lower()):
            print('Compiling '+name_prog+' in '+path2prog)
            fpc_prog(name_prog,path2prog,show_term_output=False,force_fpc=False,verbose=verbose)
        os.chdir(curr_dir)                          # Change directory back to original directory

    try:
        subprocess.check_call(cmd_list.split(), encoding='utf8', stdout=output_direct, cwd=path2prog, shell=is_windows)
    except subprocess.CalledProcessError as e:
        #don't stop if error code is 95
        if e.returncode == 95:
            # error coed 95 is a warning that at least one point did not converge
            if verbose:
                print("Error code 95")
        else:
            raise e
        # print(path2prog)
        # raise ChildProcessError
        
    os.chdir(curr_dir)                          # Change directory back to original directory




def run_multiprocess_simu(prog2run,code_name_lst,path_lst,str_lst,max_jobs=max(1,os.cpu_count()-1)):
    """run_multiprocess_simu runs simulations in parrallel (if possible) on max_jobs number of threads

    Parameters
    ----------
    prog2run : function
        name of the function that runs the simulations

    code_name_lst : list of str
        list of names of the codes to run
    
    str_lst : list of str
        List containing the strings to run for the simulations

    path_lst : list of str
        List containing the path to the folder containing the simulation program
    
    max_jobs : int, optional
        Number of threads used to run the simulations, by default os.cpu_count()-1
    """
    p = multiprocessing.Pool(max_jobs)
    results = parmap.starmap(prog2run,list(zip(code_name_lst,path_lst,str_lst)), pm_pool=p, pm_processes=max_jobs,pm_pbar=True)
    p.close()
    p.join()



def run_parallel_simu(code_name_lst,path_lst,str_lst,max_jobs=max(1,os.cpu_count()-1),verbose=False):
    """Runs simulations in parrallel on max_jobs number of threads using the
    GNU Parallel program. (https://www.gnu.org/software/parallel/). 
    If this command is used please cite:
    Tange, O. (2021, August 22). GNU Parallel 20210822 ('Kabul').
    Zenodo. https://doi.org/10.5281/zenodo.5233953

    To Install GNU Parallel on linux: (not available on Windows)
    sudo apt update
    sudo apt install parallel

    Parameters
    ----------
    prog2run : function
        name of the function that runs the simulations

    code_name_lst : list of str
        list of names of the codes to run
    
    str_lst : list of str
        List containing the strings to run for the simulations

    path_lst : list of str
        List containing the path to the folder containing the simulation program
    
    max_jobs : int, optional
        Number of threads used to run the simulations, by default os.cpu_count()-1
    """
    
    # str_lst,JV_files,Var_files,scPars_files,code_name_lst,path_lst,labels = Simulation_Inputs
    path2prog = path_lst[0]
    
    filename = 'Str4Parallel_'+str(uuid.uuid4())+'.txt'
    # tempfilepar = open(os.path.join(path2prog,filename),'w')
    with open(os.path.join(path2prog,filename),'w') as tempfilepar:
        for idx,val in enumerate(str_lst):

            str_lst[idx] = './'+code_name_lst[idx].lower() + ' ' + str_lst[idx]
            tempfilepar.write(str_lst[idx] + "\n")
    # print(str_lst)
    # tempfilepar.close()
    
    curr_dir = os.getcwd()                      # Get current directory
    os.chdir(path2prog)                         # Change directory to working directory
    log_file = os.path.join(path2prog,'logjob_'+str(uuid.uuid4()) + '.dat')
    cmd_str = 'parallel --joblog '+ log_file +' --jobs '+str(int(max_jobs))+' --bar -a '+os.path.join(path2prog,filename)
   
    if verbose:
        stdout = None
        stderr = None
    else:
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
    try:
        subprocess.run(cmd_str.split(), stdout=stdout, stderr=stdout)
        # subprocess.check_call(cmd_str.split(), encoding='utf8', cwd=path2prog, stdout=verbose) # use to run but now does not, no idea why
        # os.system("your_command") # also an option
    except subprocess.CalledProcessError as e:
        #don't stop if error code is 95
        if e.returncode != 0:
            log = pd.read_csv(log_file,sep='\t',usecols=['Exitval'],error_bad_lines=False)
            if log['Exitval'].isin([0, 95]).all():
                os.remove(log_file)
                if verbose:
                    print("Error code 95")
            else:
                print(log['Exitval'])
                raise e

    os.chdir(curr_dir)                          # Change directory back to original directory




def RunSimulation(Simulation_Inputs,max_jobs=os.cpu_count()-2 ,do_multiprocessing=True,verbose=False):
    """RunSimulation runs the simulation code using the inputs provided by PrepareSimuInputs function.

    Parameters
    ----------
    Simulation_inputs : list
        needed inputs for the simulations, see PrepareSimuInputs function for an example of the input
    max_jobs : int, optional
        number of parallel thread to run the simulations, by default os.cpu_count()-2
    do_multiprocessing : bool, optional
        whether to do multiprocessing when possible , by default True
    verbose : bool, optional
        Display text message, by default False
    """    
    str_lst,JV_files,Var_files,scPars_files,code_name_lst,path_lst,labels = Simulation_Inputs
    
    # Check if parallel is available
    if do_multiprocessing and (shutil.which('parallel') == None):
        do_multiprocessing = False
        if verbose:
            print('GNU Parallel not found, running simulations sequentially')
            print('Please install GNU Parallel (https://www.gnu.org/software/parallel/)')

    # Run simulation
    if do_multiprocessing and (len(str_lst) > 1):
        run_parallel_simu(code_name_lst,path_lst,str_lst,int(max_jobs),verbose=verbose)
        # run_multiprocess_simu(run_code,code_name_lst,path_lst,str_lst,int(max_jobs)) #old way of running the simulations for some reason in Ubuntu 22.04 it stopped working, worked on Ubuntu until 20.10 so you test it in case you do not want to use parallel 
        
    else:
        if verbose:
            show_progress = True
        else:
            show_progress = False
        for i in range(len(str_lst)):
            run_code(code_name_lst[i],path_lst[i],str_lst[i],show_term_output=show_progress)

def PrepareSimuInputs(Path2Simu,parameters=None,fixed_str=None,CodeName = 'SimSS',verbose=False):
    """Prepare the command strings to run and the fixed parameters and output files
    This procedure will make command with fixed_str and all the possible combinations of the input parameters in parameters. (see itertools.product)

    Parameters
    ----------
    Path2Simu : str
        path to the folder containing  the simulation program.
    parameters : list, optional
        list of dictionaries containing the parameters to simulate, by default None
        Structure example: [{'name':'Gfrac','values':list(np.geomspace(1e-3,5,3))},{'name':'mun_0','values':list(np.geomspace(1e-8,1e-7,3))}]
    fixed_str : _type_, optional
        Add any fixed string to the simulation command, by default None
    CodeName : str, optional
        code name, can be ['SimSS','simss','ZimT','zimt'], by default 'SimSS'
    verbose : bool, optional
        Verbose?, by default False

    Returns
    -------
    list
        list of lists containing the command strings to run the simulations, the output files and the fixed parameters
        str_lst, JV_files, Var_files, scPars_files, code_name_lst, path_lst, labels
    """    
    ## Prepare strings to run
    # Fixed string
    if fixed_str is None:
        if verbose:
            print('No fixed string given, using default value')
        fixed_str = ''  # add any fixed string to the simulation command

    # Parameters to vary
    if parameters is None:
        if verbose:
            print('No parameters list was given, using default value')
        parameters = []
        parameters.append({'name':'Gfrac','values':[0]})

    
    # Initialize     
    str_lst,JV_files,Var_files,scPars_files,code_name_lst,path_lst,labels,val,nam = [],[],[],[],[],[],[],[],[]

    if len(parameters) > 1:
        for param in parameters: # initalize lists of values and names
            val.append(param['values'])
            nam.append(param['name'])

        idx = 0
        for i in list(itertools.product(*val)): # iterate over all combinations of parameters
            str_line = ''
            lab = ''
            JV_name = 'JV'
            Var_name = 'Var'
            scPars_name = 'scPars'
            for j,name in zip(i,nam):
                str_line = str_line +'-'+name+' {:.2e} '.format(j)
                lab = lab+name+' {:.2e} '.format(j)
                JV_name = JV_name +'_'+name +'_{:.2e}'.format(j)
                Var_name = Var_name +'_'+ name +'_{:.2e}'.format(j)
                scPars_name = scPars_name +'_'+ name +'_{:.2e}'.format(j)
            str_lst.append(fixed_str+ ' ' +str_line+ '-JV_file '+JV_name+ '.dat -Var_file '+Var_name+'.dat -scPars_file '+scPars_name+'.dat')# -ExpJV '+JVexp_lst[idx])
            JV_files.append(os.path.join(Path2Simu , str(JV_name+ '.dat')))
            Var_files.append(os.path.join(Path2Simu , str(Var_name+ '.dat')))
            scPars_files.append(os.path.join(Path2Simu , str(scPars_name+ '.dat')))
            code_name_lst.append(CodeName)
            path_lst.append(Path2Simu)
            labels.append(lab)
            idx += 1
    elif len(parameters) == 1:
        # str_line = ''
        # lab = ''
        # JV_name = 'JV'
        # Var_name = 'Var'
        # scPars_name = 'scPars'
        name = parameters[0]['name']
        idx = 0
        for j in parameters[0]['values']:
            str_line = ''
            lab = ''
            JV_name = 'JV'
            Var_name = 'Var'
            scPars_name = 'scPars'
            str_line = '-'+name+' {:.2e} '.format(j)
            lab = name+' {:.2e} '.format(j)
            JV_name = JV_name +'_'+name +'_{:.2e}'.format(j)
            Var_name = Var_name +'_'+ name +'_{:.2e}'.format(j)
            scPars_name = scPars_name +'_'+ name +'_{:.2e}'.format(j)
            str_lst.append(fixed_str+ ' ' +str_line+ '-JV_file '+JV_name+ '.dat -Var_file '+Var_name+'.dat -scPars_file '+scPars_name+'.dat')# -ExpJV '+JVexp_lst[idx])
            JV_files.append(os.path.join(Path2Simu , str(JV_name+ '.dat')))
            Var_files.append(os.path.join(Path2Simu , str(Var_name+ '.dat')))
            scPars_files.append(os.path.join(Path2Simu , str(scPars_name+ '.dat')))
            code_name_lst.append(CodeName)
            path_lst.append(Path2Simu)
            labels.append(lab)

            
            idx += 1
    else:
        print('No parameters given')
        str_lst.append(fixed_str+ ' -JV_file JV.dat -Var_file Var.dat -scPars_file scPars.dat')# -ExpJV '+JVexp_lst[idx])
        JV_files.append(os.path.join(Path2Simu , str('JV.dat')))
        Var_files.append(os.path.join(Path2Simu , str('Var.dat')))
        scPars_files.append(os.path.join(Path2Simu , str('scPars.dat')))
        code_name_lst.append(CodeName)
        path_lst.append(Path2Simu)
        labels.append('Simulation')


    
    return str_lst,JV_files,Var_files,scPars_files,code_name_lst,path_lst,labels

def  DegradationPrepareSimuInputs(Path2Simu,parameters=None,fixed_str=None,CodeName = 'SimSS',verbose=False):
    """Prepare the command strings to run and the fixed parameters and output files
    This procedure will make command with fixed_str and all the possible combinations of the input parameters in parameters. (see itertools.product)

    Parameters
    ----------
    Path2Simu : str
        path to the folder containing  the simulation program.
    parameters : list, optional
        list of dictionaries containing the parameters to simulate, by default None
        Structure example: [{'name':'Gfrac','values':list(np.geomspace(1e-3,5,3))},{'name':'mun_0','values':list(np.geomspace(1e-8,1e-7,3))}]
    fixed_str : _type_, optional
        Add any fixed string to the simulation command, by default None
    CodeName : str, optional
        code name, can be ['SimSS','simss','ZimT','zimt'], by default 'SimSS'
    verbose : bool, optional
        Verbose?, by default False

    Returns
    -------
    list
        list of lists containing the command strings to run the simulations, the output files and the fixed parameters
        str_lst, JV_files, Var_files, scPars_files, code_name_lst, path_lst, labels
    """    
    ## Prepare strings to run
    # Fixed string
    if fixed_str is None:
        if verbose:
            print('No fixed string given, using default value')
        fixed_str = ''  # add any fixed string to the simulation command

    # Parameters to vary
    if parameters is None:
        if verbose:
            print('No parameters list was given, using default value')
        parameters = []
        parameters.append({'name':'Gfrac','values':[0]})

    
    # Initialize     
    str_lst,JV_files,Var_files,scPars_files,code_name_lst,path_lst,labels,val,nam = [],[],[],[],[],[],[],[],[]
    names = []

    if len(parameters) > 1:
        for i in parameters:
            names.append(i['name'])
            val.append(i['values'])
        
        if not all(len(l) == len(val[0]) for l in val):
            raise ValueError('Error in the input parameters list!\n The length of each list in the parameters list must be the same')
        val = np.asarray(val)
        param2run = pd.DataFrame(val.T,columns=names)

        idx = 0
        for index, row in param2run.iterrows():
            str_line = ''
            lab = ''
            JV_name = 'JV'
            Var_name = 'Var'
            scPars_name = 'scPars'
            for name in param2run.columns:
                str_line = str_line +'-'+name+' {:.2e} '.format(row[name])
                lab = lab+name+' {:.2e} '.format(row[name])
                JV_name = JV_name +'_'+name +'_{:.2e}'.format(row[name])
                Var_name = Var_name +'_'+ name +'_{:.2e}'.format(row[name])
                scPars_name = scPars_name +'_'+ name +'_{:.2e}'.format(row[name])
            str_lst.append(fixed_str+ ' ' +str_line+ '-JV_file '+JV_name+ '.dat -Var_file '+Var_name+'.dat -scPars_file '+scPars_name+'.dat')# -ExpJV '+JVexp_lst[idx])
            JV_files.append(os.path.join(Path2Simu , str(JV_name+ '.dat')))
            Var_files.append(os.path.join(Path2Simu , str(Var_name+ '.dat')))
            scPars_files.append(os.path.join(Path2Simu , str(scPars_name+ '.dat')))
            code_name_lst.append(CodeName)
            path_lst.append(Path2Simu)
            labels.append(lab)
            idx += 1
    elif len(parameters) == 1:
        # str_line = ''
        # lab = ''
        # JV_name = 'JV'
        # Var_name = 'Var'
        # scPars_name = 'scPars'
        name = parameters[0]['name']
        idx = 0
        for j in parameters[0]['values']:
            str_line = ''
            lab = ''
            JV_name = 'JV'
            Var_name = 'Var'
            scPars_name = 'scPars'
            str_line = '-'+name+' {:.2e} '.format(j)
            lab = name+' {:.2e} '.format(j)
            JV_name = JV_name +'_'+name +'_{:.2e}'.format(j)
            Var_name = Var_name +'_'+ name +'_{:.2e}'.format(j)
            scPars_name = scPars_name +'_'+ name +'_{:.2e}'.format(j)
            str_lst.append(fixed_str+ ' ' +str_line+ '-JV_file '+JV_name+ '.dat -Var_file '+Var_name+'.dat -scPars_file '+scPars_name+'.dat')# -ExpJV '+JVexp_lst[idx])
            JV_files.append(os.path.join(Path2Simu , str(JV_name+ '.dat')))
            Var_files.append(os.path.join(Path2Simu , str(Var_name+ '.dat')))
            scPars_files.append(os.path.join(Path2Simu , str(scPars_name+ '.dat')))
            code_name_lst.append(CodeName)
            path_lst.append(Path2Simu)
            labels.append(lab)
    else:
        print('No parameters given')
        str_lst.append(fixed_str+ ' -JV_file JV.dat -Var_file Var.dat -scPars_file scPars.dat')# -ExpJV '+JVexp_lst[idx])
        JV_files.append(os.path.join(Path2Simu , str('JV.dat')))
        Var_files.append(os.path.join(Path2Simu , str('Var.dat')))
        scPars_files.append(os.path.join(Path2Simu , str('scPars.dat')))
        code_name_lst.append(CodeName)
        path_lst.append(Path2Simu)
        labels.append('Simulation')

    return str_lst,JV_files,Var_files,scPars_files,code_name_lst,path_lst,labels

# if __name__ == '__main__':
    
#     System = platform.system()                  # Operating system
#     is_windows = (System == 'Windows')          # Check if we are on Windows
    
#     if is_windows:
#         run_code('SimSS','c:/Users/lecor/Desktop/GitHub/PVLC/codes/Simulation_program/SIMsalabim_v425/SimSS',str2run='-L 110e-9')
#     else:
#         run_code('SimSS','/mnt/c/Users/lecor/Desktop/GitHub/PVLC/codes/Simulation_program/SIMsalabim_v425/SimSS',str2run='-L 110e-9')
