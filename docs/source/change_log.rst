Change Log
================================

v1.0.10
**********************
(VMLC-PV)
- Added the inverse_loss_function to recover the real objective from the models. This is needed to recover the real MSE when calculating the posterior probability.
- Added the possibility to have a custom scaling factor (p0m) for the FitParams object. This is useful when using parameters constraints in Ax which does not allow for very small values of the parameters when writing inequalities. (see  `https://github.com/facebook/Ax/issues/2183 <https://github.com/facebook/Ax/issues/2183>`_ )
- Because of the previous point the posterior probability functions have been updated to include the scaling factor.
- Added Download_SIMsalabim.py to the SIMsalabim_utils folder to download the latest version of SIMsalabim from the repository.
- Added the `basinhopping <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html>`_ and `dual_annealing <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html#scipy.optimize.dual_annealing>`_ optimizers from scipy to the optimizer.py file.
- Added the possibility to have Global stopping strategies in the optimization_botorch.py file. 
- Small typo and docstring fixes.

v1.0.9
**********************
(VMLC-PV)
- Change in optimization_botorch.py to make sure we do not run any posterior calculations unless necessary, i.e. is optimization is not MOO and if show_posterior is set to True. Also make sure that the best set of parameters is returned from the tried ones and not by using the surrogate model.
- Change in SIMsalabim_utils/RunSim.py to make sure we clean up the output files after each run.
- Small fixes in optimizer.py.
