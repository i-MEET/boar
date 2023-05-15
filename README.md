# BOAR : Bayesian Optimization for Automated Research

## Authors
[Larry Lüer](https://github.com/larryluer) \
[Vincent M. Le Corre](https://github.com/VMLC-PV)

## Institution
Institute Materials for Electronics and Energy Technology (i-MEET) \
University of Erlangen-Nuremberg (FAU) \
Erlangen, Germany

## Description
This repository contains the code to run the BOAR. BOAR is a Bayesian optimization procedure that can be use for two objectives:
1. to optimize the parameters of a simulation to fit experimental data.
2. to optimize the processing conditions in a self-driving experimental set-up.
BOAR is based on the [scikit-learn](https://scikit-learn.org/stable/) and uses the [skopt.optimizer](https://scikit-optimize.github.io/stable/modules/generated/skopt.Optimizer.html) to perform a Bayesian Optimization on a surrogate model of the objective function constructed with Gaussian Process Regression.

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
        ├── test                         # Contains the codes for testing BOAR
    └── README.md


## Installation
### With pip
To install the BOAR, you need to clone the repository and install the requirements. The requirements can be installed with the following command:

```
pip install -r requirements.txt
```

### With conda
To install the BOAR, you need to clone the repository and install the requirements. The requirements can be installed with the following command:

```
conda create -n boar 
conda activate boar
conda install --file requirements.txt
```

Note: If you plan on using all the i-MEET repositories you can use the requirements_main.txt file to install all the requirements at once.

## Additional necessary installs for the agents
### Drift-diffusion agent
The drift-diffusion agent uses the [SIMsalabim](https://github.com/kostergroup/SIMsalabim) package to run the drift-diffusion simulations.
If you plan on using the drift-diffusion agent, you need to install SIMsalabim. Note that the drift-diffusion agent can only be used on Linux to run the simulations in parallel and it has not been tested on Windows. However, all the other agents and functionalities of the BOAR can be used on Windows.

#### SIMsalabim
All the details to install SIMsalabim are detailed in the [GitHub repository](https://github.com/kostergroup/SIMsalabim). To make sure that you are running the latest version of SIMsalabim, check regularly the repository.

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