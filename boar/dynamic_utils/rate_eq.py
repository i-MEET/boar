######################################################################
############## Functions to solve rate equations #####################
######################################################################
# Version 0.1
# (c) Larry Lueer, Vincent M. Le Corre, i-MEET 2021-2023

# Package imports
import numpy as np
from scipy.integrate import solve_ivp, odeint
from functools import partial
# ignore warnings
import warnings
# warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

def Bimolecular_Trapping_equation(ktrap, kdirect, t, Gpulse, tpulse, ninit=[0],  equilibrate=True, eq_limit=1e-2, maxcount=1e3, solver_func = 'solve_ivp', **kwargs):
    """Solve the bimolecular trapping equation :\\
    
    dn/dt = G - ktrap * n - kdirect * n^2
   

    Parameters
    ----------
    t : ndarray of shape (n,)
        array of time values

    G :  ndarray of shape (n,)
        array of values of the charge carrier generation rate m^-3 s^-1

    ktrap : float
        trapping rate constant

    kdirect : float
        Bimolecular/direct recombination rate constant

    tpulse : ndarray of shape (n,), optional
        array of time values for the pulse time step in case it is different from t, by default None

    ninit : list of floats, optional
        initial value of the charge carrier density, by default [0]
    
    equilibrate : bool, optional
        make sure equilibrium is reached?, by default True
    
    eq_limit : float, optional
        relative change of the last time point to the previous one, by default 1e-2
    
    maxcount : int, optional
        maximum number of iterations to reach equilibrium, by default 1e3
    
    solver_func : str, optional
        solver function to use can be ['odeint','solve_ivp'], by default 'solve_ivp'

    kwargs : dict
        additional keyword arguments for the solver function
        'method' : str, optional
            method to use for the solver, by default 'RK45'
        'rtol' : float, optional
            relative tolerance, by default 1e-3
    
    Returns
    -------
    ndarray of shape (n,)
        array of values of the charge carrier density m^-3

    """   
    # check solver function
    if solver_func not in ['odeint','solve_ivp']:
        warnings.warn('solver function not recognized, using odeint')
        solver_func = 'odeint'

    # kwargs
    method = kwargs.get('method', 'RK45')
    rtol = kwargs.get('rtol', 1e-6)

    # check if the pulse time step is different from the time vector
    if tpulse is None:
        tpulse = t

    def dndt(t, y, tpulse, Gpulse, ktrap, kdirect):
        """Bimolecular trapping equation
        """  
        gen = np.interp(t, tpulse, Gpulse) # interpolate the generation rate at the current time point
        
        S = gen - ktrap * y - kdirect * y**2
        return S.T

    # Solve the ODE
    if equilibrate: # make sure the system is in equilibrium 
        # to be sure we equilibrate the system properly we need to solve the dynamic equation over the full range of 1/fpu in time
        rend = 1e-20 # last time point
        RealChange = 1e19 # initialize the relative change with a high number
        rstart = ninit[0]+rend
        count = 0
        while np.any(abs(RealChange) > eq_limit) and count < maxcount:
            if solver_func == 'odeint':
                r = odeint(dndt, rstart, tpulse, args=(tpulse, Gpulse, ktrap, kdirect), tfirst=True, **kwargs)
                RealChange = (r[-1] -rend)/rend # relative change of mean
                rend = r[-1] # last time point
            elif solver_func == 'solve_ivp':
                # r = solve_ivp(dndt, [t[0], t[-1]], rstart, args=(tpulse, Gpulse, ktrap, kdirect), method = method, rtol=rtol)
                r = solve_ivp(partial(dndt,tpulse = tpulse, Gpulse = Gpulse, ktrap = ktrap, kdirect = kdirect), [t[0], t[-1]], ninit, t_eval = t, method = method, rtol=rtol)
    
                RealChange  = (r.y[:,-1] -rend)/rend # relative change of mean
                rend = r.y[:,-1] # last time point
            rstart = ninit[0]+rend
            count += 1

    else:
        rstart = ninit[0]
    
    # solve the ODE again with the new initial conditions with the equilibrated system and the original time vector
    Gpulse_eq = np.interp(t, tpulse, Gpulse) # interpolate the generation rate at the current time point
    if solver_func == 'odeint':
        r = odeint(dndt, rstart, t, args=(t, Gpulse_eq, ktrap, kdirect), tfirst=True, **kwargs)
        return r[:,0]
    elif solver_func == 'solve_ivp':
        # r = solve_ivp(dndt, [t[0], t[-1]], rstart, t_eval = t, args=(t, Gpulse_eq, ktrap, kdirect), method = method, rtol=rtol)
        r = solve_ivp(partial(dndt,tpulse = t, Gpulse = Gpulse_eq, ktrap = ktrap, kdirect = kdirect), [t[0], t[-1]], rend + ninit[0], t_eval = t, method = method, rtol=rtol)
    
        return r.y[0]




def Bimolecular_Trapping_Detrapping_equation(ktrap, kdirect, kdetrap, Bulk_tr, p_0, t, Gpulse, tpulse, ninit=[0,0,0], equilibrate=True, eq_limit=1e-2,maxcount=1e3, solver_func = 'odeint',**kwargs):
    """Solve the bimolecular trapping and detrapping equation :

    dn/dt = G - ktrap * n * (Bulk_tr - n_t) - kdirect * n * (p + p_0)
    dn_t/dt = k_trap * n * (Bulk_tr - n_t) - kdetrap * n_t * (p + p_0)
    dp/dt = G - kdetrap * n_t * (p + p_0) - kdirect * n * (p + p_0)

    Parameters
    ----------
    ktrap : float
        Trapping rate constant in m^3 s^-1
    kdirect : float
        Bimolecular/direct recombination rate constant in m^3 s^-1
    kdetrap : float
        Detrapping rate constant in m^3 s^-1
    Bulk_tr : float
        Bulk trap density in m^-3
    p_0 : float
        Ionized p-doping concentration in m^-3
    t : ndarray of shape (n,)
        time values in s
    Gpulse : ndarray of shape (n,)
        array of values of the charge carrier generation rate m^-3 s^-1
    tpulse : ndarray of shape (n,), optional
        time values for the pulse time step in case it is different from t, by default None
    ninit : list of floats, optional
        initial electron, trapped electron and hole concentrations in m^-3, by default [0,0,0]
    equilibrate : bool, optional
        whether to equilibrate the system, by default True
    eq_limit : float, optional
        limit for the relative change of the last time point to the previous one to consider the system in equilibrium, by default 1e-2
    maxcount : int, optional
        maximum number of iterations to reach equilibrium, by default 1e3
    solver_func : str, optional
        solver function to use can be ['odeint','solve_ivp'], by default 'odeint'
    kwargs : dict
        additional keyword arguments for the solver function
        'method' : str, optional
            method to use for the solver, by default 'RK45'
        'rtol' : float, optional
            relative tolerance, by default 1e-3

    Returns
    -------
    ndarray of shape (n,)
        electron concentration in m^-3
    ndarray of shape (n,)
        trapped electron concentration in m^-3
    ndarray of shape (n,)
        hole concentration in m^-3
    """   

    # check solver function
    if solver_func not in ['odeint','solve_ivp']:
        warnings.warn('solver function not recognized, using odeint')
        solver_func = 'odeint'

    # kwargs
    method = kwargs.get('method', 'RK45')
    rtol = kwargs.get('rtol', 1e-3)

    # check if the pulse time step is different from the time vector
    if tpulse is None:
            tpulse = t
    
    def rate_equations(t, n, tpulse, Gpulse, ktrap, kdirect, kdetrap, Bulk_tr, p_0):
            """Rate equation of the BTD model (PEARS) 

            Parameters
            ----------
            t : float
                time in s
            n : list of floats
                electron, trapped electron and hole concentrations in m^-3
            Gpulse : ndarray of shape (n,)
                array of values of the charge carrier generation rate m^-3 s^-1
            tpulse : ndarray of shape (n,), optional
                array of time values for the pulse time step in case it is different from t, by default None
            ktrap : float
                trapping rate constant in m^3 s^-1
            kdirect : float
                Bimolecular/direct recombination rate constant in m^3 s^-1
            kdetrap : float
                detrapping rate constant in m^3 s^-1
            Bulk_tr : float
                bulk trap density in m^-3
            p_0 : float
                ionized p-doping concentration in m^-3

            Returns
            -------
            list
                Fractional change of electron, trapped electron and hole concentrations at times t
            """

            gen = np.interp(t, tpulse, Gpulse) # interpolate the generation rate at the current time point
            
            n_e, n_t, n_h = n
            
            B = kdirect * n_e * (n_h + p_0)
            T = ktrap * n_e * (Bulk_tr - n_t)
            D = kdetrap * n_t * (n_h + p_0)
            dne_dt = gen - B - T
            dnt_dt = T - D
            dnh_dt = gen - B - D
            return [dne_dt, dnt_dt, dnh_dt]

    # Solve the ODE
    if equilibrate: # equilibrate the system
        # to be sure we equilibrate the system properly we need to solve the dynamic equation over the full range of 1/fpu in time 
        rend = [1e-20,1e-20,1e-20] # initial conditions
        rstart = [rend[0] + ninit[0], rend[1] + ninit[1], rend[2] + ninit[2]] # initial conditions for the next integration
        RealChange = 1e19 # initialize the relative change with a high number
        count = 0
        while np.any(abs(RealChange) > eq_limit) and count < maxcount:

            if solver_func == 'solve_ivp':
                r = solve_ivp(partial(rate_equations,tpulse = tpulse, Gpulse = Gpulse, ktrap = ktrap, kdirect = kdirect, kdetrap = kdetrap, Bulk_tr = Bulk_tr, p_0 = p_0), [t[0], t[-1]], rstart, t_eval = None, method = method, rtol= rtol) # method='LSODA','RK45'
                # monitor only the electron concentration           
                RealChange  = (r.y[0,-1] - rend[0])/rend[0] # relative change of mean
                rend = [r.y[0,-1], r.y[1,-1], r.y[2,-1]] # last time point
            elif solver_func == 'odeint':
                r = odeint(rate_equations, rstart, tpulse, args=(tpulse, Gpulse, ktrap, kdirect, kdetrap, Bulk_tr, p_0), tfirst=True, rtol=rtol)
                RealChange = (r[-1,0]-rend[0])/rend[0] # relative change of mean
                rend = [r[-1,0], r[-1,1], r[-1,2]] # last time point

            rstart = [rend[0] + ninit[0], rend[1] + ninit[1], rend[2] + ninit[2]] # initial conditions for the next integration
            count += 1
    else:
        rstart = ninit


    # solve the ODE again with the new initial conditions with the equilibrated system and the original time vector
    Gpulse_eq = np.interp(t, tpulse, Gpulse) # interpolate the generation rate at the current time point
    if solver_func == 'solve_ivp':
        r = solve_ivp(partial(rate_equations,tpulse = t, Gpulse = Gpulse_eq, ktrap = ktrap, kdirect = kdirect, kdetrap = kdetrap, Bulk_tr = Bulk_tr, p_0 = p_0), [t[0], t[-1]], rstart, t_eval = t, method = method, rtol= rtol) # method='LSODA','RK45'
        n_e = r.y[0]
        n_t = r.y[1]
        n_h = r.y[2]
    elif solver_func == 'odeint':
        r = odeint(rate_equations, rstart, t, args=(t, Gpulse_eq, ktrap, kdirect, kdetrap, Bulk_tr, p_0), tfirst=True, rtol=rtol)
        n_e = r[:,0]
        n_t = r[:,1]
        n_h = r[:,2]

    return n_e, n_t, n_h




        


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # add boar to the path
    import sys
    sys.path.append('../..')
    from boar.dynamic_utils.pump import get_flux_density, gaussian_pump, square_pump
    
    P = 0.0039  # real Power in W
    fpu = 1000 # pump frequency in Hz
    pulse_width = 0.2*(1/fpu) # pulse width in s
    wl = 850 # pump wavelength in nm
    A = 0.3*0.3 *1e-4 # pump area in m^-2 (a rough guess)
    alpha = 1e-5 * 1e-2  # pump penetration depth in m (a rough guess) 

    flux,density = get_flux_density(P,wl,fpu,A,alpha)

#     # Define the time array
#     t = np.linspace(0, 0.98*1/fpu, 1000)

#     G = square_pump(t, fpu, pulse_width, density)
#     # Define the trapping rate
#     ktrap = 1e5

#     # Define the bimolecular recombination rate
#     kdirect = 1e-18 # m^-3 s^-1 
#     line_types = ['-', '--', '-.', ':']
#     plt.figure(0)
#     for idx1, ktrap in enumerate([1e4,1e5,1e6,1e7]):
#         for idx, kdirect in enumerate([1e-17,1e-18,1e-19]):

#             # Solve the bimolecular trapping equation
#             n = Bimolecular_Trapping_equation(ktrap, kdirect, t, G )

#             # Plot the results
#             plt.plot(t, n, label = 'ktrap = {} 1/s, kdirect = {} m^-3 s^-1'.format(ktrap,kdirect), linestyle = line_types[idx1])

#     plt.figure(1)

#     t = np.geomspace(1e-10, 0.98*1/fpu, 100000)
#     # add 0 at the beginning of the time array
#     t = np.insert(t, 0, 0)
#     G = gaussian_pump(t, fpu, 5e-9,  density, 10e-9, background=0)
#     for idx1, ktrap in enumerate([1e4,1e5,1e6,1e7]):
#         for idx, kdirect in enumerate([1e-17,1e-18,1e-19]):

#             # Solve the bimolecular trapping equation
#             n = Bimolecular_Trapping_equation(ktrap, kdirect, t, G, )

#             # Plot the results
            
#             plt.semilogx(t, n/max(n), label = 'ktrap = {} 1/s, kdirect = {} m^-3 s^-1'.format(ktrap,kdirect), linestyle = line_types[idx1])
#     plt.semilogx(t, G/max(G), 'k',label = 'pulse')

    plt.figure(2)
    t = np.geomspace(1e-10, 0.98*1/fpu, 100000)
    t = np.geomspace(1e-10, 0.98*1/fpu, 5000)
    # add 0 at the beginning of the time array
    t = np.insert(t, 0, 0)
    P = 0.039  # real Power in W
    flux,density = get_flux_density(P,wl,fpu,A,alpha)
    G = gaussian_pump(t, fpu, 5e-9,  density/100, 5e-9, background=0)
    ktrap = 3e-12
    kdirect = 4e-16
    kdetrap = 8e-15
    Bulk_tr = 1e20
    p_0 = 0e18
    import matplotlib.cm as cm
    # colors = cm.virisdis(10)
    # import viridis
    viridis = cm.get_cmap('viridis', 10)
    count = 0
    for kdetrap in [8e-15]:
        for kdirect in [4e-16]:
            n_e, n_t, n_h = Bimolecular_Trapping_Detrapping_equation(ktrap, kdirect, kdetrap, Bulk_tr, p_0, t, G, t, equilibrate = True, eq_limit = 1e-3)
            # Relation between charge carrier concentration and TRPL signal
            trpl = n_e * (n_h + p_0)
            # Take into account normalization and background
            # signal_trpl = 1e-5 * trpl / (n_e[0] * (n_h[0] + p_0))
            signal_trpl = kdirect * 0.8e-31 * n_e * (n_h + p_0)
            plt.loglog(t, signal_trpl, 'b',label = 'TRPL')
            signal_trpl = 1e-24 * (1e-2*n_e + n_h + p_0)
            # find idx max of the signal
            idx_max = np.argmax(signal_trpl)
            # find the time of the max
            t_max = t[idx_max]
            plt.loglog(t, signal_trpl, 'k',label = 'TRMC')
            # plt.plot(t-t_max, signal_trpl/max(signal_trpl),label = 'TRPL', color = viridis(count))
        count += 1
    plt.xlim(5e-10,5e-5)
    plt.ylim(1e-7,10)
    plt.savefig('test.png', dpi = 300)
    plt.show()