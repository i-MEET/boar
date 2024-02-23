Installation
================================

With pip
**********************
To install BOAR with pip you have two options:
1. Install BOAR using the `PyPI repository <https://pypi.org/project/boar-pv/>`_

.. code-block:: bash

    pip install boar-pv


2. Install BOAR using the GitHub repository
First, you need to clone the repository and install the requirements. The requirements can be installed with the following command:

.. code-block:: bash

    pip install -r requirements.txt

Similarly to the conda installation, if you plan on using the BoTorch/Ax optimizer you need to use the requirements_torch_CPU.txt file or install pytorch with the correct version for your system with the requirements.txt file.


With conda
**********************
To install the BOAR, you need to clone the repository and install the requirements. The requirements can be installed with the following command:

.. code-block:: bash

    conda create -n boar 
    conda activate boar
    conda install --file requirements.txt

if you want you can also clone your base environment by replacing the first line with:

.. code-block:: bash

    conda create -n boar --clone base


Additional necessary for BOAR with BoTorch/Ax
****************************************************************
If you plan on using only the scikit-learn optimizer, you can remove use the ''requirements.txt'' file, however if you plan on using the `Ax & BoTorch <https://ax.dev/tutorials/multiobjective_optimization.html>`_ optimizer, you have two options:

1. If you only want to use the CPU just install the requirements_torch_CPU.txt as described previously.
2. If you also want to use the GPU and use CUDA you need to refer to the pytorch website to install the correct version of pytorch for your system. First start by intalling the''requirements.txt'' file as described previously then go to the 'Get Started' section `https://pytorch.org/get-started/locally/ <https://ax.dev/tutorials/multiobjective_optimization.html>`_ and select the correct option for your system. You should have to run a command similar to the following one:

.. code-block:: bash

    pip install ax-platform
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117


Additional necessary installs for the agents
============================================
Drift-diffusion agent
***********************
The drift-diffusion agent uses the `SIMsalabim <https://github.com/kostergroup/SIMsalabim>`_ package to run the drift-diffusion simulations.
If you plan on using the drift-diffusion agent, you need to install SIMsalabim. Note that the drift-diffusion agent can only be used on Linux to run the simulations in parallel and it has not been tested on Windows. However, all the other agents and functionalities of the BOAR can be used on Windows.

SIMsalabim
#######################
All the details to install SIMsalabim are detailed in the `GitHub repository <https://github.com/kostergroup/SIMsalabim>`_. To make sure that you are running the latest version of SIMsalabim, check regularly the repository.

Parallel simulations
#######################
On Linux, you have the option to run the simulations in parrallel. Sadly, not on windows (yet).
To be able to run simulations in parrallel efficiently and with any threading or multiprocessing from python we use the `parallel` from the `GNU prallel <https://www.gnu.org/software/parallel/>`_ project.
To install on linux run:

.. code-block:: bash

    sudo apt update
    sudo apt install parallel

You can also use `Anaconda <https://anaconda.org/>`_:

.. code-block:: bash

    conda install -c conda-forge parallel

To test is the installation worked by using by running the following command in the terminal:

.. code-block:: bash

    parallel --help


Disclaimer
================================
This repository is still under development. If you find any bugs or have any questions, please contact us.