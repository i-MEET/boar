Change Log
================================

v1.0.9
**********************
(VMLC-PV)
- Change in optimization_botorch.py to make sure we do not run any posterior calculations unless necessary, i.e. is optimization is not MOO and if show_posterior is set to True. Also make sure that the best set of parameters is returned from the tried ones and not by using the surrogate model.
- Change in SIMsalabim_utils/RunSim.py to make sure we clean up the output files after each run.
- Small fixes in optimizer.py.
