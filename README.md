# BOAR : Bayesian Optimization for Automated Research

## Authors
[Vincent M. Le Corre](https://github.com/VMLC-PV)  
[Larry Lüer](https://github.com/larryluer)


## Institution
Institute Materials for Electronics and Energy Technology (i-MEET)  
University of Erlangen-Nuremberg (FAU)  
Erlangen, Germany  

## Description
This repository contains the code to run the BOAR. BOAR is a Bayesian optimization procedure that can be use for two objectives:
1. to optimize the parameters of a simulation to fit experimental data.
2. to optimize the processing conditions in a self-driving experimental set-up.  

BOAR has two main optimizer, one based on [scikit-learn](https://scikit-learn.org/stable/) which uses the [skopt.optimizer](https://scikit-optimize.github.io/stable/modules/generated/skopt.Optimizer.html) and one based on the [BoTorch](https://botorch.org/) and [Ax](https://ax.dev/) packages. 

## Repository Folder Structure
    .
    ├── Main                             # Main directory, place Notebooks here to run them
        ├── boar                         # Main directory for the BOAR codes
            ├── core                     # Directory with the multi-objective optimization codes
            ├── agents                   # Directory with the different experimental agents
            ├── dynamic_utils            # Directory with the different utility functions to solve rate equations
            ├── SIMsalabim_utils         # Directory with the different utility functions to run SIMsalabim, the drift-diffusion simulator
        ├── Notebooks                    # Contains clean versions of the Notebooks, Notebooks need to be moved to the main directory to be used         
        ├── Example_Data                 # Contains some example data for the notebooks
        ├── SIMsalabim                   # Contains the codes to run the drift-diffusion simulations using SIMsalabim
        ├── test                         # Contains the codes for testing BOAR
    └── README.md


## Installation
### With pip
To install BOAR with pip you have two options:
1. Install BOAR using the [PyPI repository](https://pypi.org/project/boar-pv/)
```
    pip install boar-pv
```

2. Install BOAR using the GitHub repository
First, you need to clone the repository and install the requirements. The requirements can be installed with the following command:
```
    pip install -r requirements.txt
```
Similarly to the conda installation, if you plan on using the BoTorch/Ax optimizer you need to use the requirements_torch_CPU.txt file or install pytorch with the correct version for your system with the requirements.txt file.

### With conda
To install the BOAR, you need to clone the repository and install the requirements. The requirements can be installed with the following command:

```
conda create -n boar 
conda activate boar
conda install --file requirements.txt
```
if you want you can also clone your base environment by replacing the first line with:
```
conda create -n boar --clone base
```

If you plan on using only the scikit-learn optimizer, you can remove use the ''requirements.txt'' file, however if you plan on using the BoTorch/Ax optimizer, you have two options:

1. If you only want to use the CPU just install the requirements_torch_CPU.txt file:
2. If you also want to use the GPU and use CUDA you need to refer to the pytorch website to install the correct version of pytorch for your system. First start by intalling the''requirements.txt'' file as described previously then go to the 'Get Started' section [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select the correct option for your system. You should have to run a command similar to the following one:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

## Additional necessary for BOAR with BoTorch/Ax
If you plan on using only the scikit-learn optimizer, you can remove use the ''requirements.txt'' file, however if you plan on using the [Ax & BoTorch](https://ax.dev/tutorials/multiobjective_optimization.html) optimizer, you have two options:

1. If you only want to use the CPU just install the requirements_torch_CPU.txt file as described previously.
2. If you also want to use the GPU and use CUDA you need to refer to the pytorch website to install the correct version of pytorch for your system. First start by intalling the''requirements.txt'' file as described previously then go to the 'Get Started' section [https://pytorch.org/get-started/locally/](https://ax.dev/tutorials/multiobjective_optimization.html) and select the correct option for your system. You should have to run a command similar to the following one:

```
    pip install ax-platform
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

## Additional necessary installs for the agents
### Drift-diffusion agent
The drift-diffusion agent uses the [SIMsalabim](https://github.com/kostergroup/SIMsalabim) package to run the drift-diffusion simulations.
If you plan on using the drift-diffusion agent, you need to install SIMsalabim. Note that the drift-diffusion agent can only be used on Linux to run the simulations in parallel and it has not been tested on Windows. However, all the other agents and functionalities of the BOAR can be used on Windows.

#### SIMsalabim
All the details to install SIMsalabim are detailed in the [GitHub repository](https://github.com/kostergroup/SIMsalabim). To make sure that you are running the latest version of SIMsalabim, check regularly the repository.  
Note that we include SIMsalabim in this repository as a submodule, however, you are free to use a different verion.  
If you use SIMsalabim, please follow the instruction on the [GitHub repository](https://github.com/kostergroup/SIMsalabim) to cite the package.

#### Parallel simulations
On Linux, you have the option to run the simulations in parrallel. Sadly, not on windows (yet).
To be able to run simulations in parrallel efficiently and with any threading or multiprocessing from python we use the `parallel` from the [GNU prallel](https://www.gnu.org/software/parallel/) project.
To install on linux run:
```
sudo apt update
sudo apt install parallel
```
You can also use [Anaconda](https://anaconda.org/):
```
conda install -c conda-forge parallel
```
To test is the installation worked by using by running the following command in the terminal:
```
parallel --help
```
It is also possible to use the `parmap` package to run the simualtions in parallel. To switch, use the `run_multiprocess_simu` instead of `run_parallel_simu` in the `RunSimulation` function in `/VLC_units/Simulation/RunSim.py` folder. However, this does not work on Ubuntu version 22.04 but seems to work on older versions.


## Disclaimer
This repository is still under development. If you find any bugs or have any questions, please contact us.