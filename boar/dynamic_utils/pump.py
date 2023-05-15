######################################################################
############### Define the pump style for transient ##################
################           experiments              ##################
######################################################################
# Version 0.1
# (c) Larry Lueer, Vincent M. Le Corre, i-MEET 2021-2023

# Package imports
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import trapz
from scipy.signal import square
from scipy import constants


## Physics constants
q = constants.value(u'elementary charge')
eps_0 = constants.value(u'electric constant')
kb = constants.value(u'Boltzmann constant in eV/K')
c = constants.value(u'speed of light in vacuum')
h_eV = constants.value(u'Planck constant in eV s')
h_J = constants.value(u'Planck constant')


def get_flux_density(P,wvl,fpu,A,alpha):
    """From the measured power, repetition rate and area, 
    get photons/m2 and approximate photons/m3 per pulse

    Parameters
    ----------
    P : float
        total CW power of pulse in W
    wvl : float
        excitation wavelength in nm
    fpu : float
        repetition rate in s^-1
    A : float
        effective pump area in m^-2
    alpha : float
        penetration depth in m

    Returns
    -------
    flux : float
        flux in photons m^-2
    density : float
        average volume density in photons m^-3
    """    
    
    E = h_J*c/(wvl*1e-9) # convert wavelength to J for a single photon
    Epu = P/fpu # energy in J of a single pulse
    Nph = Epu/E # Photon number in pulse
    flux = Nph/A # flux in photons m^-2
    density = flux/alpha # average absorbed density in photons m^-3
    return flux,density
    
    
    
def square_pump(t, fpu, pulse_width, P, t0 = 0, background=0,Gfrac=1):
    """Square pump pulse

    Parameters
    ----------
    t : ndarray of shape (n,)
        array of time values in seconds

    fpu : float
        pump frequency in Hz

    pulse_width : float
        width of the pump pulse in seconds

    P : float
        total volume density of generated photons m^-3
    
    t0 : float, optional
        time shift of the pump pulse, by default 0
    
    background : float, optional
        background volume density of generated photons, by default 0
    
    Gfrac : float, optional
        scaling for the power of the pulse, by default 1

    Returns
    -------
    ndarray of shape (n,)
        density of generated photons m^-3 at each time point

    """ 

    #convert pulse_width to fraction of the period
    pulse_width = pulse_width / (1/fpu) 
    
    pump = 0.5*square(2 * np.pi * fpu * (t-t0), pulse_width) + 0.5 # pump pulse
    putot = trapz(pump,t) # total pump power
    pump = pump / putot * P * Gfrac # normalize the pump pulse to the total pump power
    pump = pump + background # add background

    return pump

def gaussian_pulse_norm(t, tpulse, width):
    """Returns a gaussian pulse

    Parameters
    ----------
    t : 1-D sequence of floats
        t time axis (unit: s)
    tpulse : float
        tpulse center of the pulse (unit: s)
    width : float
        width of the pulse (unit: s)

    Returns
    -------
    1-D sequence of floats
        Vector containing the gaussian pulse
    """    
    return np.exp(-np.power(t - tpulse, 2.) / (2 * np.power(width, 2.)))

def gaussian_pump(t, fpu,  pulse_width, P, t0, background=0, Gfrac=1):
    """Gaussian pump pulse

    Parameters
    ----------
    t : ndarray of shape (n,)
        array of time values in seconds
    
    fpu : float
        pump frequency in Hz
    
    pulse_width : float
        width of the pump pulse in seconds
    
    P : float
        total volume density of generated photons m^-3
    
    t0 : float
        center of the pulse in seconds

    background : float, optional
        background volume density of generated photons, by default 0

    Gfrac : float, optional
        scaling for the power of the pulse, by default 1

    Returns
    -------
    ndarray of shape (n,)
        density of generated photons m^-3 at each time point
    """  
    # find time step in t
    max_dt = 0
    for i in range(len(t)-1):
        dt = t[i+1]-t[i]
        if dt > max_dt:
            max_dt = dt
    # check if the pulse is smaller than the time step
    if pulse_width < max_dt:
        # raise ValueError('The pulse width is smaller than the time step. Increase the pulse width or decrease the time step.')
        # raise warning if the pulse is smaller than the time step
        warnings.warn('The pulse width is smaller than the max time step. If you are using s linear time step you need to increase the pulse width or decrease the time step. If you are using non-linear time step make sure that you have small enough time step around the pulse or some pulses might not appear.')

    if max(t) > 1/fpu:
        # number of pulses
        Np = int(max(t) * fpu)
        # time axis for the pulses
        tp = np.linspace(0, 1/fpu, int(1e4))
        # pump pulse
        pp = gaussian_pulse_norm(tp, 0.5*1/fpu +t0, pulse_width)
        # total pump power
        putot = trapz(pp,tp)
        # normalize the pump pulse to the total pump power
        pp = pp / putot * P * Gfrac
        # add the pulses
        for i in range(Np+1):
            if i == 0:
                pump = np.interp(t + 0.5*1/fpu , tp + i*1/fpu , pp) 
            else: 
                pump = pump + np.interp(t + 0.5*1/fpu , tp + i*1/fpu , pp) 
    else:

        pump = gaussian_pulse_norm(t , t0, pulse_width) # pump pulse

        putot = trapz(pump,t) # total pump power

        pump = pump / putot * P * Gfrac# normalize the pump pulse to the total pump power

     


    pump = pump + background # add background

    return pump


def pump_from_file(t, filename, P = None, background=0, Gfrac = 1, sep=None):
    """Pump pulse from file

    Parameters
    ----------
    t : ndarray of shape (n,)
        array of time values in seconds

    filename : str
        path to the file containing the pump pulse
    
    P : float, optional
        total volume density of generated photons m^-3, by default None

    background : float, optional
        background volume density of generated photons, by default 0

    Gfrac : float, optional
        scaling for the power of the pulse, by default 1

    sep : str, optional
        delimiter to use, by default None

    Returns
    -------
    ndarray of shape (n,)
        density of generated photons m^-3 at each time point
    """    
    if sep is None:
        data = pd.read_csv(filename, delim_whitespace=True, names=['t', 'pump'])
    else:
        data = pd.read_csv(filename, sep=sep, names=['t', 'pump'])

    pump = np.interp(t, data['t'], data['pump'])


    if P is not None:
        pump = np.interp(t, data['t'], data['pump'])
        baseline = np.min(pump)
        pump = pump - baseline # remove baseline
        putot = trapz(pump,t) # total pump power
        pump = pump / putot * P * Gfrac# normalize the pump pulse to the total pump power
        pump = pump + background # add background
    
    else:
        pump = pump + background # add background

    return pump

def initial_carrier_density(t, fpu, N0, background = 0, Gfrac = 1):
    """Initial carrier density

    Parameters
    ----------
    t : ndarray of shape (n,)
        array of time values in seconds
    
    fpu : float
        pump frequency in Hz
    
    N0 : float
        initial carrier density in m^-3
    
    background : float, optional
        background carrier density, by default 0

    Gfrac : float, optional
        scaling for the power of the pulse, by default 1    
    

    Returns
    -------
    ndarray of shape (n,)
        initial carrier density in m^-3
    """   
    n = np.zeros(len(t))
    # repeat the initial carrier density every 1/fpu
    count = 1
    n[0] = N0*Gfrac
    for idx, tt in enumerate(t):
        if idx == 0:
            continue
        if tt >= count/fpu and t[idx-1] <= count/fpu:
            n[idx] = N0 * Gfrac
            count += 1
    

    n = n + background 
    return n
    # if max(t) > 1/fpu:
    #     # number of pulses
    #     Np = int(max(t) * fpu)
    #     # time axis for the pulses
    #     tp = np.linspace(0, 1/fpu, int(1e4))
    #     idx = (np.abs(tp - 0.5/fpu)).argmin()
    #     pp = np.zeros(len(tp))
    #     pp[idx] = 1
    #     # total pump power
    #     putot = trapz(pp,tp)
    #     # normalize the pump pulse to the total pump power
    #     pp = pp / putot * N0
    #     # add the pulses
    #     count = 1
    #     pump = np.zeros(len(t))

    #     for idx, tt in enumerate(t):
    #         if idx == 0:
    #             pump[idx] = max(pp)
    #             continue
    #         if tt >= count/fpu and t[idx-1] <= count/fpu:
    #             pump[idx] = max(pp)
    #             count += 1

    # else:
    #     pump = np.zeros(len(t))
    #     pump[0] = 1
  
    #     putot = trapz(pump,t) # total pump power

    #     pump = pump / putot * N0 # normalize the pump pulse to the total pump power

    # return pump + background






if __name__ == "__main__":
    import matplotlib.pyplot as plt
#     P = 0.0039  # real Power in W
#     fpu = 10000 # pump frequency in Hz
#     pulse_width = 0.2*(1/fpu) # pulse width in s
#     wl = 850 # pump wavelength in nm
#     A = 0.3*0.3 *1e-4 # pump area in m^-2 (a rough guess)
#     alpha = 1e-5 * 1e-2  # pump penetration depth in m (a rough guess) 

#     flux,density = get_flux_density(P,wl,fpu,A,alpha)

#     # Test the square pump pulse
#     # plt.figure(0)
#     # t = np.linspace(0, 1/fpu, 1000)
#     # plt.plot(t, square_pump(t, fpu, pulse_width, density, t0=1e-5, background=0))
#     # plt.plot(t, square_pump(t, fpu, pulse_width, density, t0=0, background=0))
#     # plt.plot(t, square_pump(t, fpu, pulse_width, density, t0=0, background=1e28))

#     plt.figure(2)
    
#     # plt.semilogx(t, gaussian_pump(t, fpu, pulse_width/2, 20e-7, density, background=0))
#     # plt.semilogx(t, gaussian_pump(t, fpu, pulse_width/3, 10e-7, density, background=0))
#     fpu = 1000000
#     t = np.geomspace(1e-10, 0.98*1/fpu, 100000)
#     # add 0 at the beginning of the time array
#     t = np.insert(t, 0, 0)
#     pulse_width = 5e-9
#     plt.plot(t, gaussian_pump(t, fpu, pulse_width, density, pulse_width*2,  background=0))
#     plt.plot(t, gaussian_pump(t, fpu, pulse_width, density, pulse_width*3,  background=0))
#     plt.plot(t, gaussian_pump(t, fpu, pulse_width, density, pulse_width*2,  background=1e29))

#     plt.figure(3)
#     t = np.linspace(0, 4.5/fpu, int(1e6))
#     plt.plot(t, gaussian_pump(t, fpu, pulse_width, density, pulse_width*2,  background=0))
#     t = np.linspace(0, 4/fpu, 1000)
#     plt.plot(t, gaussian_pump(t, fpu, pulse_width, density, pulse_width*3, background=0))
#     t = np.linspace(0, 1/fpu, 1000)
#     plt.plot(t, gaussian_pump(t, fpu, pulse_width, density, pulse_width*2, background=0))

    plt.figure(4)
    fpu = 1000 # pump frequency in Hz
    background = 1e20 # background pump density in m^-3
    # create a pump pulse
    t = np.linspace(0,2.99*1/fpu,200)
    N0 = 1e21
    plt.semilogy(t, initial_carrier_density(t, fpu, N0, background=background))
    # plt.ylim(1e19, 1e22)

    plt.show()

