####################################################################################
##################### Download SIMsalabim from GitHub ##############################
####################################################################################
# Author: Vincent M. Le Corre
# Github: https://github.com/VMLC-PV


# Description: This script contains a function to download SIMsalabim from GitHub and compile it
# Import libraries
import json,requests,os,git,shutil,zipfile,io,asyncio,time


def download_simsalabim(path2prog=None,verbose=True):
    """Download SIMsalabim from GitHub and extract the files to the current working directory

    Parameters
    ----------
    path : str, optional
        Path to the directory where SIMsalabim will be downloaded, by default None\\
        If None, SIMsalabim will be downloaded to the current working directory in a folder named 'SIMsalabim'
    verbose : bool, optional
        Print the download progress, by default True

    """ 
    if verbose:
        print('Downloading SIMsalabim from GitHub')
        print('For more information, please visit:')
        print('https://github.com/kostergroup/SIMsalabim')
        print('')
        print('This may take a few seconds...')
        print('Please wait...')

    cwd = os.getcwd() # Get current working directory   
    if path2prog is None:
        path2prog = os.path.join(cwd, 'SIMsalabim')

    folders = []
    folder_name = 'kostergroup-SIMsalabim-'
    for dirpath, dirnames, files in os.walk(cwd):
        for dirname in dirnames:
            if dirname.startswith(folder_name):
                overwrite = None
                overwrite = input(f'SIMsalabim is already downloaded, do you want to overwrite {path2prog}? (y/n): ')
                while overwrite not in ['y','n']:
                    print('Please enter y or n')
                    overwrite = input(f'SIMsalabim is already downloaded, do you want to overwrite {path2prog}? (y/n): ')

                if overwrite == 'y':
                    shutil.rmtree(os.path.join(cwd,dirname))
                    print(f"Found a folder named {dirname}")


    if os.path.exists(path2prog):
        
        overwrite = None
        overwrite = input(f'SIMsalabim is already downloaded, do you want to overwrite {path2prog}? (y/n): ')
        while overwrite not in ['y','n']:
            print('Please enter y or n')
            overwrite = input(f'SIMsalabim is already downloaded, do you want to overwrite {path2prog}? (y/n): ')

        if overwrite == 'y':
            # Rename folder
            shutil.rmtree(path2prog)
            # # Get the files from the latest release
            url = 'https://api.github.com/repos/kostergroup/SIMsalabim/zipball'
            response = requests.get(url)

            # Open the zip file
            z = zipfile.ZipFile(io.BytesIO(response.content))

            # Extract all the files
            z.extractall(path=cwd)

            for dirpath, dirnames, files in os.walk(cwd):
                for dirname in dirnames:
                    if dirname.startswith(folder_name):
                        # Rename folder
                        shutil.move(os.path.join(cwd, dirname), path2prog)
                        break
        else:
            print(' We are keeping the current SIMsalabim version')

            

    else:
        # # Get the files from the latest release
        url = 'https://api.github.com/repos/kostergroup/SIMsalabim/zipball'
        response = requests.get(url)

        # Open the zip file
        z = zipfile.ZipFile(io.BytesIO(response.content))

        # Extract all the files
        z.extractall(path=cwd)

        for dirpath, dirnames, files in os.walk(cwd):
            for dirname in dirnames:
                if dirname.startswith(folder_name):
                    # print(f"Found a folder named {dirname}")
                    # Rename folder
                    shutil.move(os.path.join(cwd, dirname), path2prog)
                    break


    try :
        # Check if fpc is installed
        fpc_version = os.popen('fpc -iV').read()
        #remove dots
        fpc_version = fpc_version.replace('.','')
        # check if fpc version is 3.2.0 or higher
        if int(fpc_version) < 320:
            raise Exception('fpc version is lower than 3.2.0, please update fpc to version 3.2.0 or higher/n for now we are downloading the pre-compiled binaries from GitHub')
        
        print('fpc is installed so we are compiling the SIMsalabim programs')
        # Compile the programs
        # compile simss with fpc in os.path.join(path2prog,'SimSS')
        cwd = os.getcwd() # Get current working directory
        os.chdir(os.path.join(path2prog,'SimSS'))
        os.system('fpc simss.pas')
        os.chdir(cwd)

        # compile zimt with fpc in os.path.join(path2prog,'ZimT')
        cwd = os.getcwd() # Get current working directory
        os.chdir(os.path.join(path2prog,'ZimT'))
        os.system('fpc zimt.pas')
        os.chdir(cwd)

        if verbose:
            print('SIMsalabim programs have been compiled successfully!')

    except:
        if verbose:
            print('')
            print('fpc is not installed so we are skipping the compilation of the SIMsalabim programs')
            print('For now we are downloading the pre-compiled binaries from GitHub')
            print('')
            print('If the binaries do not work or if you want to compile the SIMsalabim programs yourself, please install fpc.')
            print('For more information, please visit: https://www.freepascal.org/')
            print('')
        # # Get the assets from the latest release
        url = "https://api.github.com/repos/kostergroup/SIMsalabim/releases/latest"
        response = requests.get(url)
        data = json.loads(response.text)


        for asset in data["assets"]:
            download_url = asset["browser_download_url"]
            filename = asset["name"]
            response = requests.get(download_url)
            open(os.path.join(cwd,filename), "wb").write(response.content)


        for dirpath, dirnames, files in os.walk(cwd):

            for filename in files:
                if filename.startswith('simss') and os.path.exists(os.path.join(cwd, filename)):
                    # print(f"Found a folder named {filename}")
                    # Rename folder
                    shutil.move(os.path.join(cwd, filename), os.path.join(cwd, 'SIMsalabim','SimSS',filename))
                elif filename.startswith('zimt') and os.path.exists(os.path.join(cwd, filename)):
                    # print(f"Found a folder named {filename}")
                    # Rename folder
                    shutil.move(os.path.join(cwd, filename), os.path.join(cwd, 'SIMsalabim','ZimT',filename))
                else:
                    pass
    if verbose:
        print('')
        print('SIMsalabim has been downloaded successfully!')
        print('')

if __name__ == '__main__':
    
    download_simsalabim() # Download SIMsalabim

    
             


