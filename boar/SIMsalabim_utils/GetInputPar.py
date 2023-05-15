##########################################################################
#################### Read and get parameters from    #####################
#################### dev_para file and command string ####################
##########################################################################
# Author: Vincent M. Le Corre
# Github: https://github.com/VMLC-PV

# Import libraries
import os

def GetParFromStr(str2run):
    """Get parameters from command string for SIMsalabim or ZimT

    Parameters
    ----------
    str2run : STR
        Command string for SIMsalabim or ZimT

    Returns
    -------
    dict
        Contains the parameters and values from the command string
    """    

    str2run = ' '.join(str2run.split()) #remove extra white space

    str2run = str2run.split()

    Names= str2run[::2]
    Values = str2run[1::2]
    ParStrDic = {}
    for i,j in enumerate(Names):
        Names[i] = Names[i].replace('-', '')
        # Values[i] = float(Values[i])
        try: # if the value is a float
            ParStrDic[Names[i]] = float(Values[i])
        except: # if the value is a string
            ParStrDic[Names[i]] = Values[i]
    return ParStrDic

def ReadParameterFile(path2file):
    """Get all the parameters from the 'Device_parameters.txt' file
    for SIMsalabim and ZimT
    Parameters
    ----------
    path2file : str
        Path to the 'Device_parameters.txt'

    Returns
    -------
    dict
        Contains the parameters and values from the 'Device_parameters.txt'
    """    
    
    lines = []
    ParFileDic = {}
    with open(path2file) as f:
        lines = f.readlines()

    count = 0
    for line in lines:
        line = line.replace(' ', '')
        if line[0] != '*' and (not line.isspace()):
            equal_idx = line.find('=')
            star_idx = line.find('*')
            # print(line[0:equal_idx] , line[equal_idx+1:star_idx])
            ParFileDic[line[0:equal_idx] ] = line[equal_idx+1:star_idx]
            count += 1
            # print(f'line {count}: {line}')   
    return ParFileDic

def ChosePar(parname,ParStrDic,ParFileDic):
    """Chose if we use parameter from 'Device_parameters.txt'
    or from the command string for SIMsalabim and ZimT

    Parameters
    ----------
    parname : str
        Parameter name as defined in 'Device_parameters.txt' file
    ParStrDic : dict
        Contains the parameters and values from the command string
    ParFileDic : dict
        Contains the parameters and values from the 'Device_parameters.txt'

    Returns
    -------
    str
        String of the parameter value (need to be converted to float if needed)
    """    
    if parname in ParStrDic.keys():
        parval = ParStrDic[parname]
    else :
        parval = ParFileDic[parname]
    
    return parval