######################################################################
####################### Cleaning output files ########################
######################################################################
# Author: Vincent M. Le Corre
# Github: https://github.com/VMLC-PV

# Import libraries
import os,subprocess

def clean_up_output(filename_start,path):
    """Delete output files from the simulation

    Parameters
    ----------
    filename_start : str
        string containing the begining of the filename to delete

    path : str
        path to the directory where we clean the output
    """ 
    for fname in os.listdir(path):
        if fname.startswith(filename_start) and not os.path.isdir(os.path.join(path,fname)):
            os.remove(os.path.join(path,fname))

def Store_output_in_folder(filenames,folder_name,path):
    """Move output files from the simulation into new folder

    Parameters
    ----------
    filenames : list of str
        list of string containing the name of the files to move

    folder_name : str
        name of the folder where we store the output files
        
    path : str
        directory of the folder_name (creates one if it does not already exist)
    """    

    # Create directory if it does not exist
    if not os.path.exists(os.path.join(path,folder_name)):
        os.makedirs(os.path.join(path,folder_name))
    # move file into the new folder
    for i in filenames:
        if os.path.exists(os.path.join(path,i)):
            os.replace(os.path.join(path,i),os.path.join(path,folder_name,i))
        else:
            print('File {} does not exist'.format(os.path.join(path,i)))

def clean_file_type(ext,path):
    """Delete files of a given type in the current directory

    Parameters
    ----------
    ext : str
        extension of the files to delete
    
    path : str
        path to the directory where we clean the output

    """ 
    for fname in os.listdir(path):
        if fname.endswith(ext):
            os.remove(os.path.join(path,fname))

