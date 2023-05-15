##########################################################################
#################### Create device_parameter file    #####################
#######################  for SimSS and ZimT  ######### ####################
##########################################################################
# Author: Vincent M. Le Corre
# Github: https://github.com/VMLC-PV

# Import libraries
import os,re,sys,shutil


def CheckProgVersion(path2file):
    """Check for the program version number in device_parameters.txt

    Parameters
    ----------
    path2file : str
        path to the device_parameters.txt file
    """    
    
    lines = []
    ParFileDic = {}
    with open(path2file) as f:
        lines = f.readlines()
    
    for i in lines:
        try:
            if 'version:' in i:
                str_line = re.sub(r"\s+", "", i, flags=re.UNICODE)
                idx = str_line.find(':')
                version = float(str_line[idx+1:])
        except ValueError:
            sys.exit("Version number is not in the file or is not a float, execution is stopped.\nPlease review the device_parameters.txt file in folder:\n"+path2file)
    if version < 4.33:
        sys.exit('You are running an older version of SIMsalabim which has not been tested with this version of the code. This may lead to issues.\nPlease update your SIMsalabim version to 4.33 or higher.\nExecution is stopped.')
    return(version)

def MakeDevParFileCopy(path2file,path2file_copy):
    """Make a copy of the file 'path2file in 'path2file_copy

    Parameters
    ----------
    path2file : str
        absolute path to the simulation folder that contains the device_parameters.txt file.
    path2file_copy : _type_
        absolute path to the simulation folder that will contain the device_parameters_old.txt file.
    """    
    shutil.copyfile(path2file,path2file_copy)

def UpdateDevParFile(ParFileDic,path2file,MakeCopy=True):
    """Update the device_parameters.txt with the values contained in ParFileDic
    Has to be used with SIMsalabim v4.33 or higher

    Parameters
    ----------
    ParFileDic : dic
        Dictioanry containing the values to be written in the file.
    path2file : str
        absolute path to the simulation folder that contains the device_parameters.txt file. 
    MakeCopy : bool, optional
        Make a copy of the previous device_parameters.txt file in device_parameters_old.txt, by default True
    """    

    # Saves a copy of the original device_parameters.txt file
    if MakeCopy:
        path2file_old = os.path.join(path2file,'device_parameters_old.txt')
        path2file = os.path.join(path2file,'device_parameters.txt')
        MakeDevParFileCopy(path2file,path2file_old)
    else:
        path2file = os.path.join(path2file,'device_parameters.txt')
    
    # Check for version number
    CheckProgVersion(path2file)

    with open(path2file) as f: # read lines in device_parameters.txt
        lines = f.readlines()
    newlines = []
    

    for i in lines:
        str_line = re.sub(r"\s+", "", i, flags=re.UNICODE)

        if str_line.startswith('*') or len(str_line) == 0: # line is a comment
            newlines.append(i)
        else:
            idx1 = str_line.find('=') # index of '='
            idx2 = str_line.find('*') # index of '*'

            i_start = i.find('=')
            i_end = i.find('*')
            # if len(idx):
            str1 = i[:i_start+1]
            str2 = i[i_end:] 
            if str_line[:idx1] in ParFileDic.keys():
                newlines.append(str1 + ' ' + str(ParFileDic[str_line[:idx1]]) + ' '*10 + str2)
            else:
                newlines.append(i)
    with open(path2file, 'w') as f: # write new lines in device_parameters.txt
        for line in newlines:
            f.write(line)


# if __name__ == '__main__':
    # path2file = os.path.join(os.getcwd(),'device_parameters.txt')
    # path2file_copy = os.path.join(os.getcwd(),'device_parameters_old.txt')
    # MakeDevParFileCopy(path2file,path2file_copy)
    # UpdateDevParFilev433({'T':'298','mun_0':'1e-8'},'/mnt/c/Users/lecor/Desktop/GitHub/PVLC_v2/Simulation_program/SIMsalabimv433_SCLC/SimSS',MakeCopy=True)
