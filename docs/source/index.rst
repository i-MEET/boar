.. boar documentation master file, created by
   sphinx-quickstart on Thu Jun 29 13:46:02 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BOAR's documentation!
================================
**BOAR** is a Python package for Bayesian Optimization. It is designed to be a flexible and easy-to-use tool for researchers and practitioners alike. While it can be used for a wide variety of applications, the implemented agents are specifically designed for solar cell and general research on semiconductor. The package is designed to be modular and extensible, allowing users to easily add new agents.

Most of the agents in BOAR can be used to fit a model to experimental data, yet BOAR can be also used to optimize experimental process condition and guide experimentation in an automated fashion.

See the `BOAR GitHub repository <https://github.com/i-MEET/boar>`_ for more information.

Authors
==================
* `Larry Lüer <https://github.com/larryluer>`_
* `Vincent M. Le Corre <https://github.com/VMLC-PV>`_

Institution
==================
.. figure:: ../logo/imeet_logo_square.jpg
   :align: left
   :width: 100px
   :alt: i-MEET logo

| `Institute Materials for Electronics and Energy Technology <https://www.i-meet.ww.uni-erlangen.de/>`_ (i-MEET)
| University of Erlangen-Nuremberg (FAU) 
| Erlangen, Germany

|

What is currently implemented in BOAR? : The optimizers & agents
#################################################################

The optimizers
**********************
The optimizers are used to find the optimal parameters for a given optimization problem.

Available optimizers are:

* `scikit-optimize <https://scikit-optimize.github.io/stable/index.html
* `scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_
* `Ax & BoTorch <https://ax.dev/tutorials/multiobjective_optimization.html>`_ 

Currently, the main optimizer implemented in BOAR is based on  `scikit-optimize <https://scikit-optimize.github.io/stable/index.html>`_ (skopt). It is the default optimizer used in BOAR and the most tested one.

For more information on the optimizers, see the optimizer section.

The agents
**********************
The agents are used to run a model depending on the input parameters given by the user or the optimizers. Users are free to create their own agents. For more information on the agents, see the agents section.

Available agents are:

* Drift-diffusion agent (DD)
* Transient absorption spectroscopy agent (TAS)
* Transient photoluminescence agent (TrPL)
* Transient microwave conductivity agent (TrMC)
* Transfer matrix agent (TM)

.. toctree::
   :maxdepth: 3
   :caption: Getting started:

   installation

.. toctree::
   :maxdepth: 3
   :caption: Model fitting examples:

   examples/DD_Fit_fake_OPV.ipynb
   examples/DD_Fit_real_OPV.ipynb
   examples/TAS_fit_BT_model_fake.ipynb
   examples/MO_TrPL_TrMC.ipynb
   gallery_fitting

.. toctree::
   :maxdepth: 3
   :caption: Optimization and DoE examples:

   examples/BOAR_Exp.ipynb
   examples/BOAR_Exp_MO.ipynb
   examples/TM_AVT_Jsc_MO_custom_func.ipynb
   gallery_Opti_DoE

.. toctree::
   :maxdepth: 4
   :caption: BOAR API:
   
   modules
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
