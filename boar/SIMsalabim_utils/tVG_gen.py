################################################################
############### Function to format tVG files ###################
################################################################
# Author: Vincent M. Le Corre
# Github: https://github.com/VMLC-PV

# Import libraries
import math,sys
import pandas as pds
import matplotlib.pyplot as plt
import numpy as np 
from scipy import integrate

def gaussian_pulse(t, tpulse, width, Imax):
    """Returns a gaussian pulse

    Parameters
    ----------
    t : 1-D sequence of floats
        t time axis (unit: s)
    tpulse : float
        tpulse center of the pulse (unit: s)
    width : float
        width of the pulse (unit: s)
    Imax : float
        Imax maximum of the pulse

    Returns
    -------
    1-D sequence of floats
        Vector containing the gaussian pulse
    """    
    return Imax *np.exp(-np.power(t - tpulse, 2.) / (2 * np.power(width, 2.)))


def zimt_light_decay(tmin,tmax,Gstart,Gfinal,Va,steps=100,trf = 20e-9,time_exp =False,tVG_name='tVG.txt'):
    """Make tVG file for light decay experiment

    Parameters
    ----------
    tmin : float
        first time step after 0 (unit: s)
    tmax : float
        final time step (unit: s)
    Gstart : float
        initial generation rate, i.e. light intensity (steady-state) (unit: m^-3 s^-1)
    Gfinal : float
        final generation rate (unit: m^-3 s^-1)
    Va : float
        applied voltage (unit: V)
    steps : float
        number of time steps
    trf : float, optional
        LED/laser fall/rise time, by default 20e-9 (unit: s)
    time_exp : bool, optional
        If True exponential time step is used, else linear time step, by default False
    tVG_name : str, optional
        tVG_file name, by default 'tVG.txt'
    """    

    # Chose between exponential or linear time step
    if time_exp == True:
        t = np.geomspace(tmin,tmax,steps)
    else :
        t = np.linspace(tmin,tmax,steps)

    t=np.insert(t,0,0)
    V,G = [],[]
    # Calculate the light decay assuming and exponential decay of the generation rate
    # with a lifetime (trf) of the LED/laser fall/rise time
    for i in t:
        G.append((Gstart-Gfinal)*np.exp(-i/trf)+Gfinal)
        V.append(Va)

    tVG = pds.DataFrame(np.stack([t,np.asarray(V),np.asarray(G)]).T,columns=['t','Vext','Gehp'])

    tVG.to_csv(tVG_name,sep=' ',index=False,float_format='%.3e')

def zimt_voltage_step(tmin,tmax,Vstart,Vfinal,Gen,steps=100,trf = 10e-9,time_exp =False,tVG_name='tVG.txt'):
    """Make tVG file for Voltage decay experiment

    Parameters
    ----------
    tmin : float
        first time step after 0 (unit: s)
    tmax : float
        final time step (unit: s)
    Vstart : float
        initial applied voltage (steady-state) (unit: V)
    Vfinal : float
        final applied voltage (unit: V)
    Gen : float
        constant generation rate (i.e. light intensity) (unit: m^-3 s^-1)
    steps : int, optional
        number of time steps, by default 100
    trf : float, optional
        Voltage pulse fall/rise time , by default 10e-9 (unit: s)
    time_exp : bool, optional
        If True exponential time step is used, else linear time step, by default False
    tVG_name : str, optional
        tVG_file name, by default 'tVG.txt'
    """ 

    # Chose between exponential or linear time step
    if time_exp == True:
        t = np.geomspace(tmin,tmax,num=steps)
    else :
        t = np.linspace(tmin,tmax,int(steps),endpoint=True)

    t=np.insert(t,0,0)
    V,G = [],[]
    # Calculate the voltage decay assuming and exponential decay of the voltage
    # with a lifetime (trf) of the voltage pulse fall/rise time
    for i in t:
        V.append((Vstart-Vfinal)*np.exp(-i/trf)+Vfinal)
        G.append(Gen)

    tVG = pds.DataFrame(np.stack([t,np.asarray(V),np.asarray(G)]).T,columns=['t','Vext','Gehp'])

    tVG.to_csv(tVG_name,sep=' ',index=False,float_format='%.3e')


def zimt_JV_sweep(Vstart,Vfinal,scan_speed,Gen,steps,time_exp =False,tVG_name='tVG.txt'):
    """Make tVG file for one JV sweep experiment

    Parameters
    ----------
    Vstart : float
        initial applied voltage (steady-state) (unit: V)
    Vfinal : float
        final applied voltage (unit: V)
    scan_speed : float
        scan speed (unit: V/s)
    Gen : float
        constant generation rate (i.e. light intensity) (unit: m^-3 s^-1)
    steps : int
        number of JV points
    time_exp : bool, optional
        If True exponential time step is used, else linear time step, by default False
    tVG_name : str, optional
        tVG_file name, by default 'tVG.txt'
    """    
    # Calculate duration of the JV sweep depending on the scan speed
    tmax = abs(Vfinal - Vstart)/scan_speed

    # Chose between exponential or linear time step
    if time_exp == True:
        t = np.geomspace(0,tmax,int(steps))
        t=np.insert(t,0,0)
    else :
        t = np.linspace(0,tmax,int(steps),endpoint=True)
    
    V,G = [],[]
    for i in t:
        V.append(scan_speed*i + Vstart)
        G.append(Gen)

    tVG = pds.DataFrame(np.stack([t,np.asarray(V),np.asarray(G)]).T,columns=['t','Vext','Gehp'])

    tVG.to_csv(tVG_name,sep=' ',index=False,float_format='%.3e')

# def zimt_JV_double_sweep_old(Vstart,Vfinal,scan_speed,Gen,steps,Vacc=-0.1,time_exp = False,tVG_name='tVG.txt'):
#     """Make tVG file for double JV sweep experiment
#     Scan voltage back and forth

#     Parameters
#     ----------
#     Vstart : float
#         initial applied voltage (steady-state) (unit: V)
#     Vfinal : float
#         final applied voltage (unit: V)
#     scan_speed : float
#         scan speed (unit: V/s)
#     Gen : float
#         constant generation rate (i.e. light intensity) (unit: m^-3 s^-1)
#     steps : int
#         number of JV points 
#     Vacc : float, optional
#         point of accumulation of row of V's, note: Vacc should be slightly larger than Vmax or slightly lower than Vmin (unit: V)
#     time_exp  : bool, optional
#         If True exponential time step is used, else linear time step, by default False
#     tVG_name : str, optional
#         tVG_file name, by default 'tVG.txt'
#     """
    
#     V,G = [],[]
#     if time_exp == True:
#         if Vacc > Vstart and Vacc < Vfinal:
#             print('/!\ Error /!\ ')
#             print('Vacc should be slightly larger than Vmax or slightly lower than Vmin')
#             print('We are stopping the program, change the value of Vacc')
#             sys.exit()
#         # Vacc = -0.1
#         d = Vacc - Vfinal
#         idx = 0
#         t = [0]
#         for i in range(int(steps/2)):
#             LogarithmicV = Vacc - d*np.exp((1-i/(int(steps/2)-1))*np.log((Vacc-Vstart)/d))
#             if abs(LogarithmicV) < 1e-10: # to correct for numerical approx around 0
#                 LogarithmicV = 0 
#             V.append(LogarithmicV)
#             if idx > 0:
#                 t.append(t[idx-1] + (abs(LogarithmicV-V[idx-1]))/scan_speed)
#             G.append(Gen)
#             idx += 1

#         Vrev = V[::-1]
#         Vrev.pop(0)
#         V.extend(Vrev)
#         for i in Vrev:
#             t.append(t[idx-1] + (abs(i-V[idx-1]))/scan_speed)
#             G.append(Gen)
#             idx += 1
#     else:
#         # Calculate duration of one sweep, the total experiment duration will be 2*tmax  
#         tmax = abs((Vfinal - Vstart)/scan_speed)
#         t1 = np.linspace(0,tmax,int(steps/2),endpoint=True)
#         t2 = np.linspace(tmax,2*tmax,int(steps/2),endpoint=True)
#         t2 = np.delete(t2,[0])
#         t = np.append(t1,t2)
#         for i in t:
#             if i < tmax:
#                 V.append(np.sign(Vfinal-Vstart)*scan_speed*i + Vstart)
#             else:
#                 V.append(-np.sign(Vfinal-Vstart)*(scan_speed*i) + np.sign(Vfinal-Vstart)*Vstart +2*Vfinal)
#             G.append(Gen)
        

#     tVG = pds.DataFrame(np.stack([t,np.asarray(V),np.asarray(G)]).T,columns=['t','Vext','Gehp'])

#     tVG.to_csv(tVG_name,sep=' ',index=False,float_format='%.3e')
def zimt_JV_double_sweep(Vstart,Vfinal,scan_speed,Gen,steps,Vacc=-0.1,time_exp = False,tVG_name='tVG.txt'):
    """Make tVG file for double JV sweep experiment
    Scan voltage back and forth

    Parameters
    ----------
    Vstart : float
        initial applied voltage (steady-state) (unit: V)
    Vfinal : float
        final applied voltage (unit: V)
    scan_speed : float
        scan speed (unit: V/s)
    Gen : float
        constant generation rate (i.e. light intensity) (unit: m^-3 s^-1)
    steps : int
        number of JV points 
    Vacc : float, optional
        point of accumulation of row of V's, note: Vacc should be slightly larger than Vmax or slightly lower than Vmin (unit: V)
    time_exp  : bool, optional
        If True exponential time step is used, else linear time step, by default False
    tVG_name : str, optional
        tVG_file name, by default 'tVG.txt'
    """
    
    V,G = [],[]
    if time_exp == True:
        if Vacc > Vstart and Vacc < Vfinal:
            print('/!\ Error /!\ ')
            print('Vacc should be slightly larger than Vmax or slightly lower than Vmin')
            print('We are stopping the program, change the value of Vacc')
            sys.exit()
        # Vacc = -0.1
        d = Vacc - Vfinal
        idx = 0
        t = [0]
        for i in range(int(steps/2)):
            LogarithmicV = Vacc - d*np.exp((1-i/(int(steps/2)-1))*np.log((Vacc-Vstart)/d))
            if abs(LogarithmicV) < 1e-10: # to correct for numerical approx around 0
                LogarithmicV = 0 
            V.append(LogarithmicV)
            if idx > 0:
                t.append(t[idx-1] + (abs(LogarithmicV-V[idx-1]))/scan_speed)
            G.append(Gen)
            idx += 1

        Vrev = V[::-1]
        Vrev.pop(0)
        V.extend(Vrev)
        for i in Vrev:
            t.append(t[idx-1] + (abs(i-V[idx-1]))/scan_speed)
            G.append(Gen)
            idx += 1
    else:
        # Calculate duration of one sweep, the total experiment duration will be 2*tmax  
        tmax = abs((Vfinal - Vstart)/scan_speed)
        t1 = np.linspace(0,tmax,int(steps/2),endpoint=True)
        t2 = np.linspace(tmax,2*tmax,int(steps/2),endpoint=True)
        t2 = np.delete(t2,[0])
        t = np.append(t1,t2)
        for i in t:
            if i < tmax:
                V.append(np.sign(Vfinal-Vstart)*scan_speed*i + Vstart)
            else:
                V.append(-np.sign(Vfinal-Vstart)*(scan_speed*(i-tmax))  + Vfinal)
            G.append(Gen)
        

    tVG = pds.DataFrame(np.stack([t,np.asarray(V),np.asarray(G)]).T,columns=['t','Vext','Gehp'])

    tVG.to_csv(tVG_name,sep=' ',index=False,float_format='%.3e')

def zimt_tdcf(tmin,tmax,Vpre,Vcol,Gen,tpulse,tstep,tdelay,width_pulse = 2e-9,tVp = 10e-9,time_exp=False,steps=100,tVG_name='tVG.txt'):
    """Make tVG file for time-delayed collection field (TDCF)

    Parameters
    ----------
    tmin : float
        first time step after 0 (unit: s)

    tmax : float
        final time step (unit: s)

    Vpre : float
        initial applied voltage (steady-state) or pre-bias in typical TDCF language (unit: V)

    Vcol : float
        final applied voltage or collection bias in typical TDCF language (unit: V)

    Gen : float
        Total number of carrier generated by the gaussian pulse (unit: m^-3)

    tpulse : float
        middle of the gaussian pulse (unit: s)

    tstep : float
        time step for the linear regime (unit: s)

    tdelay : float
        delay between middle of laser pulse and voltage switch (unit: s)

    width_pulse : float, optional
        width of the light pulse (unit: s), by default 2e-9

    tVp : float, optional
        Voltage pulse fall/rise time (unit: s), by default 10e-9

    time_exp : bool, optional
        if True chose exponential time step else keep time step linear, by default False

    steps : int, optional
        if time_exp = True number of exponential time step after voltage switch, by default 100
        
    tVG_name : str, optional
        tVG_file name, by default 'tVG.txt'
    """    
    
    tswitch = width_pulse + tdelay
    V,G = [],[]

    # Define laser pulse by a Gaussian and always keep tstep to 1e-10 for during the pulse to ensure that the amount of photon is consistent when changing tstep
    t = np.arange(tmin,width_pulse*1.5,1e-9)
    
    for i in t:
        if i < tswitch:
            V.append(Vpre)
        else:
            V.append((Vpre-Vcol)*np.exp(-i/tVp)+Vcol)
    G = gaussian_pulse(t,1.5*width_pulse/2,width_pulse,1)

    if width_pulse< tswitch:
        # time step before voltage delay 
        t1 = np.arange(width_pulse*1.5,tswitch-tstep,tstep)

        for i in t1:
            if i < tswitch:
                V.append(Vpre)
            else:
                V.append((Vpre-Vcol)*np.exp(-i/tVp)+Vcol)
            G=np.append(G,0)

        # Begin of the voltage voltage delay 
        if time_exp == True:
            t2 = np.geomspace(tswitch,tmax,num=steps)
        else :
            t2 = np.arange(tswitch,tmax,tstep)

        for i in t2:
                V.append((Vpre-Vcol)*np.exp(-i/tVp)+Vcol)
                G=np.append(G,0)
        
        t = np.append(t,t1)
        t = np.append(t,t2)
    else:
        # Begin of the voltage voltage delay 
        if time_exp == True:
            t2 = np.geomspace(1.5*width_pulse,tmax,num=steps)
        else :
            t2 = np.arange(1.5*width_pulse,tmax,tstep)

        for i in t2:
                V.append((Vpre-Vcol)*np.exp(-i/tVp)+Vcol)
                G=np.append(G,0)
        
        t = np.append(t,t2)

    # Insert initial conditions
    t = np.insert(t,0,0)
    V = np.asarray(V)
    V = np.insert(V,0,Vpre)
    G = np.asarray(G)
    G = np.insert(G,0,0)
    if Gen > 0:
        # ensure that the total number of generated charges is equal to Gen
        int_G  = integrate.cumtrapz(G, t, initial=0)
        G = G*Gen/int_G[-1]
    else:
        G = 0*G

    tVG = pds.DataFrame(np.stack([t,V,G]).T,columns=['t','Vext','Gehp'])

    tVG.to_csv(tVG_name,sep=' ',index=False,float_format='%.3e') 


def zimt_BACE(tmin,tmax,Gen,Vpre,Vextr,tLp = 20e-9,tVp = 10e-9,time_exp=False,steps=100,tVG_name='tVG.txt'):
    """Make tVG file for bias-assisted charge extraction (BACE)

    Parameters
    ----------
    tmin : float
        first time step after 0 (unit: s)
    tmax : float
        final time step (unit: s)
    Gen : float
        initial generation rate (i.e. light intensity) (unit: m^-3 s^-1)
    Vpre : float
        initial applied voltage (or pre-bias) (unit: V)
    Vextr : float
        extraction voltage (unit: V)
    tLp : float, optional
        LED pulse fall/rise time (unit: s), by default 20e-9
    tVp : float, optional
        Voltage pulse fall/rise time (unit: s), by default 10e-9
    time_exp : bool, optional
        if True chose exponential time step else keep time step linear, by default False
    steps : int, optional
        if time_exp = True number of exponential time step after voltage switch, by default 100
    tVG_name : str, optional
        tVG_file name, by default 'tVG.txt'
    """    

    V,G = [],[]
    
    if time_exp == True:
        t = np.geomspace(tmin,tmax,num=steps)
    else :
        t = np.linspace(tmin,tmax,steps)
    t=np.insert(t,0,0)
    for i in t:
            V.append((Vpre-Vextr)*np.exp(-i/tVp)+Vextr)
            G=np.append(G,(Gen)*np.exp(-i/tLp))

    tVG = pds.DataFrame(np.stack([t,np.asarray(V),np.asarray(G)]).T,columns=['t','Vext','Gehp'])

    tVG.to_csv(tVG_name,sep=' ',index=False,float_format='%.3e') 

def zimt_TPV(tmin,tmax,Gen_pulse,G0,tpulse,width_pulse = 2e-9,time_exp =False,steps=100,tVG_name='tVG.txt'):
    """Make tVG file for transient photovoltage (TPV) experiment

    Parameters
    ----------
    tmin : float
        first time step after 0 (unit: s)
    tmax : float
        final time step (unit: s)
    Gen_pulse : float
        Total number of carrier generated by the gaussian pulse (unit: m^-3)
    G0 : float
        background generation rate (i.e. light intensity) (unit: m^-3 s^-1)
    tpulse : float
        middle of the gaussian pulse (unit: s)
    width_pulse : float, optional
        width of the light pulse (unit: s), by default 2e-9
    time_exp : bool, optional
        if True chose exponential time step else keep time step linear, by default False
    steps : int, optional
        if time_exp = True number of exponential time step after voltage switch, by default 100
    tVG_name : str, optional
        tVG_file name, by default 'tVG.txt'
    """    

    # Define laser pulse by a Gaussian and always keep tstep to 1e-9 for during the pulse to ensure that the amount of photon is consistent when changing tstep
    t = np.arange(tmin,tpulse + width_pulse*3 - 1e-9,1e-9)
    t=np.insert(t,0,0)
    V,G = [],[]

    for i in t:
        V.append('oc')   
    G = gaussian_pulse(t,tpulse,width_pulse,Gen_pulse)
    # G[0] = G0 #set G = G0 at t = 0

    if Gen_pulse > 0:
        # ensure that the total number of generated charges is equal to Gen
        int_G  = integrate.cumtrapz(G, t, initial=0)
        G = G*Gen_pulse/int_G[-1]
    else:
        G = G0

    if time_exp == True:
        t1 = np.geomspace(tpulse + width_pulse*3,tmax,steps)
    else :
        t1 = np.linspace(tpulse + width_pulse*3, tmax, steps)

    for i in t1:
        G=np.append(G,G0)
        V.append('oc')
    
    t = np.append(t,t1)

    for i in range(len(G)): #set G = 0 when G is too small (for stability in ZimT)
        if G[i] < G0:
            G[i] = G0
    
    

    tVG = pds.DataFrame(np.stack([t,np.asarray(V),np.asarray(G)]).T,columns=['t','Vext','Gehp'])

    tVG.to_csv(tVG_name,sep=' ',index=False,float_format='%.3e')


def zimt_TPC_Gauss_pulse(tmin,tmax,Gen_pulse,G0,tpulse,width_pulse = 2e-9,time_exp =False,steps=100,tVG_name='tVG.txt'):
    """Make tVG file for transient photocurrent (TPC) experiment with a gaussian pulse for the light excitation.

    Parameters
    ----------
    tmin : float
        first time step after 0 (unit: s)
    tmax : float
        final time step (unit: s)
    Gen_pulse : float
        Total number of carrier generated by the gaussian pulse (unit: m^-3)
    G0 : float
        background generation rate (i.e. light intensity) (unit: m^-3 s^-1)
    tpulse : float
        middle of the gaussian pulse (unit: s)
    width_pulse : float, optional
        width of the light pulse (unit: s), by default 2e-9
    time_exp : bool, optional
        if True chose exponential time step else keep time step linear, by default False
    steps : int, optional
        if time_exp = True number of exponential time step after voltage switch, by default 100
    tVG_name : str, optional
        tVG_file name, by default 'tVG.txt'
    """    

    # Define laser pulse by a Gaussian and always keep tstep to 1e-9 for during the pulse to ensure that the amount of photon is consistent when changing tstep
    t = np.arange(tmin,tpulse + width_pulse*3 - 1e-9,1e-9)
    t=np.insert(t,0,0)
    V,G = [],[]

    for i in t:
        V.append(0)   
    G = gaussian_pulse(t,tpulse,width_pulse,Gen_pulse)
    # G[0] = G0 #set G = G0 at t = 0

    if Gen_pulse > 0:
        # ensure that the total number of generated charges is equal to Gen
        int_G  = integrate.cumtrapz(G, t, initial=0)
        G = G*Gen_pulse/int_G[-1]
    else:
        G = G0

    if time_exp == True:
        t1 = np.geomspace(tpulse + width_pulse*3,tmax,steps)
    else :
        t1 = np.linspace(tpulse + width_pulse*3, tmax, steps)

    for i in t1:
        G=np.append(G,G0)
        V.append(0)
    
    t = np.append(t,t1)

    for i in range(len(G)): #set G = 0 when G is too small (for stability in ZimT)
        if G[i] < G0:
            G[i] = G0
    
    

    tVG = pds.DataFrame(np.stack([t,np.asarray(V),np.asarray(G)]).T,columns=['t','Vext','Gehp'])

    tVG.to_csv(tVG_name,sep=' ',index=False,float_format='%.3e')

def zimt_TPC_square_pulse(tmin,tmax,Gen_pulse,G0,width_pulse,tLp = 20e-9,time_exp =False,steps=100,tVG_name='tVG.txt'):
    """Make tVG file for transient photocurrent (TPC) experiment with a gaussian pulse for the light excitation.

    Parameters
    ----------
    tmin : float
        first time step after 0 (unit: s)
    tmax : float
        final time step (unit: s)
    Gen_pulse : float
        Generation rate at the top of the squared pulse __--__ (i.e. light intensity) (unit: m^-3 s^-1)
    G0 : float
        background generation rate (i.e. light intensity) (unit: m^-3 s^-1)
    width_pulse : float
        width of the light pulse (unit: s)
    tLp : float, optional
        LED pulse fall/rise time (unit: s), by default 20e-9
    time_exp : bool, optional
        if True chose exponential time step else keep time step linear, by default False
    steps : int, optional
        if time_exp = True number of exponential time step after voltage switch, by default 100
    tVG_name : str, optional
        tVG_file name, by default 'tVG.txt'
    """    

    V,G = [],[]

    if time_exp == True:
        t = np.geomspace(tmin,width_pulse,int(steps/2))
        t1 = np.geomspace(tmin,tmax-width_pulse,int(steps/2))
        t = np.append(t,t1+width_pulse)
    else :
        t = np.linspace(tmin, tmax, steps)

    
    V,G = [],[]
    for i in t:
        if i < width_pulse:
            if (Gen_pulse)*(1-np.exp(-i/tLp))+G0 > 1e10: #set G = 0 when G is too small (for stability in ZimT)
                G.append((Gen_pulse)*(1-np.exp(-i/tLp))+G0)
            else:
                G.append(0)
        else:
            if Gen_pulse*np.exp(-(i-width_pulse)/tLp)+G0 > 1e10:
                G.append(Gen_pulse*np.exp(-(i-width_pulse)/tLp)+G0)
            else:
                G.append(0)
        V.append(0)
    
    #set starting condition for t = 0, G = G0
    t=np.insert(t,0,0)
    V = np.asarray(V)
    G = np.asarray(G)
    V=np.insert(V,0,0)
    G=np.insert(G,0,G0)

    tVG = pds.DataFrame(np.stack([t,V,G]).T,columns=['t','Vext','Gehp'])

    tVG.to_csv(tVG_name,sep=' ',index=False,float_format='%.7e')


def zimt_CELIV(tmin,tmax,Voffset,slopeV,Gen,tpulse,tstep,tdelay,width_pulse = 2e-9,time_exp=False,steps=100,tVG_name='tVG.txt'):
    """Make tVG file for charge extraction by linearly increasing voltage (CELIV) experiment

    Parameters
    ----------
    tmin : float
        first time step after 0 (unit: s)

    tmax : float
        final time step (unit: s)

    Voffset : float
        initial applied voltage (steady-state) (unit: V)

    slopeV : float
        slope of the applied voltage increase (V/s)

    Gen : float
        Total number of carrier generated by the gaussian pulse (unit: m^-3)

    tpulse : float
        middle of the gaussian pulse (unit: s)

    tstep : float
        time step for the linear regime (unit: s)

    tdelay : float
        delay between middle of laser pulse and voltage switch (unit: s)

    width_pulse : float, optional
        width of the light pulse (unit: s), by default 2e-9

    time_exp : bool, optional
        if True chose exponential time step else keep time step linear, by default False

    steps : int, optional
        if time_exp = True number of exponential time step after voltage switch, by default 100

    tVG_name : str, optional
        tVG_file name, by default 'tVG.txt'
    """    
    
    tswitch = tpulse + tdelay
    V,G = [],[]

    # Define laser pulse by a Gaussian and always keep tstep to 1e-9 for during the pulse to ensure that the amount of photon is consitent when changing tstep
    t = np.arange(tmin,tpulse + width_pulse*3 - 1e-9,1e-9)
    

    for i in t:
        if i < tswitch:
            V.append(Voffset)
        else:
            V.append(slopeV*(i-tswitch)+Voffset)
        G=np.append(G,gaussian_pulse(i,tpulse,width_pulse,Gen))
    

    if tpulse + width_pulse*3 < tswitch:
        # time step before voltage delay 
        t1 = np.arange(tpulse + width_pulse*3,tswitch-tstep,tstep)

        for i in t1:
            if i < tswitch:
                V.append(Voffset)
            else:
                V.append(slopeV*(i-tswitch)+Voffset)
            G=np.append(G,0)

        # Begin of the voltage voltage delay 
        if time_exp == True:
            t2 = np.geomspace(tswitch,tmax,num=steps)
        else :
            t2 = np.arange(tswitch,tmax,tstep)

        for i in t2:
                V.append(slopeV*(i-tswitch)+Voffset)
                G=np.append(G,0)
        
        t = np.append(t,t1)
        t = np.append(t,t2)
    else:
        # Begin of the voltage voltage delay 
        if time_exp == True:
            t2 = np.geomspace(tpulse + width_pulse*3,tmax,num=steps)
        else :
            t2 = np.arange(tpulse + width_pulse*3,tmax,tstep)

        for i in t2:
                V.append(slopeV*(i-tswitch)+Voffset)
                G=np.append(G,0)
        
        t = np.append(t,t2)


    # Insert initial conditions
    t = np.insert(t,0,0)
    V = np.asarray(V)
    V = np.insert(V,0,Voffset)
    G = np.asarray(G)
    G = np.insert(G,0,0)

    if Gen > 0:
        # ensure that the total number of generated charges is equal to Gen
        int_G  = integrate.cumtrapz(G, t, initial=0)
        G = G*Gen/int_G[-1]
    else:
        G = 0*G

    tVG = pds.DataFrame(np.stack([t,np.asarray(V),np.asarray(G)]).T,columns=['t','Vext','Gehp'])

    tVG.to_csv(tVG_name,sep=' ',index=False,float_format='%.3e')

def zimt_impedance(Vapp,Vamp,freq,Gen,steps=100,tVG_name='tVG.txt'):
    """Make tVG file for impedance experiment


    Parameters
    ----------
    Vapp : [type]
        offset applied voltage (steady-state) (unit: V)
    Vamp : [type]
        amplitude of the voltage perturbation (unit: V) 
    freq : [type]
        frequency of the oscillation (unit: Hz)
    Gen : [type]
        max generation rate (i.e. light intensity) of the gaussian pulse (unit: m^-3 s^-1)
    steps : int, optional
        number of time step, by default 100
    tVG_name : str, optional
        tVG_file name, by default 'tVG.txt'
    """    

    w = 2*math.pi*freq
    t = np.linspace(0,3/freq,steps)

    G,V = [],[]
    for i in t:
        G.append(Gen)
        V.append(Vapp + Vamp * np.sin(w*i))
 

    tVG = pds.DataFrame(np.stack([t,np.asarray(V),np.asarray(G)]).T,columns=['t','Vext','Gehp'])

    tVG.to_csv(tVG_name,sep=' ',index=False,float_format='%.3e')


def zimt_TrPL(tmin,tmax,Gen,G0,Vapp,tstep,tpulse,width_pulse = 2e-9,time_exp =False,steps=100,tVG_name='tVG.txt'):
    """Make tVG file for transient photovoltage (TPV) experiment

    Parameters
    ----------
    tmin : float
        first time step after 0 (unit: s)

    tmax : float
        final time step (unit: s)

    Gen : float
        Total number of carrier generated by the gaussian pulse (unit: m^-3)

    G0 : float
        background generation rate (i.e. light intensity) (unit: m^-3 s^-1)
    
    Vapp : float
        Applied voltage (unit: V)
    
    tstep : float
        time step for the linear regime (unit: s)
    
    tpulse : float
        middle of the gaussian pulse (unit: s)
    
    width_pulse : float, optional
        width of the light pulse (unit: s), by default 2e-9
    
    time_exp : bool, optional
        if True chose exponential time step else keep time step linear, by default False
    
    steps : int, optional
        if time_exp = True number of exponential time step after voltage switch, by default 100
    
    tVG_name : str, optional
        tVG_file name, by default 'tVG.txt'
    """    

     # Define laser pulse by a Gaussian and always keep tstep to 1e-9 for during the pulse to ensure that the amount of photon is consitent when changing tstep
    t = np.arange(tmin,tpulse + width_pulse*3 - 1e-9,1e-9)
    
    V,G = [],[]

    for i in t:
        V.append(Vapp)   
    G = gaussian_pulse(t,tpulse,width_pulse,Gen)

    if time_exp == True:
        t1 = np.geomspace(tpulse + width_pulse*3,tmax,steps)
    else :
        t1 = np.arange(tpulse + width_pulse*3,tmax,tstep)

    for i in t1:
        G=np.append(G,0)
        V.append(Vapp)
    
    t = np.append(t,t1)

    # Insert initial conditions
    t = np.insert(t,0,0)
    V = np.asarray(V)
    V = np.insert(V,0,Vapp)
    G = np.asarray(G)
    G = np.insert(G,0,0)

    # ensure that the total number of generated charges is equal to Gen
    int_G  = integrate.cumtrapz(G, t, initial=0)
    G = G*Gen/int_G[-1]

    if G0 > 0:
        for i in range(len(G)): #set G = G0 as background illumination, including during the light pulse
            if G[i] < G0:
                G[i] = G[i] + G0
            if G[i] < 1:
                G[i] = 0


    tVG = pds.DataFrame(np.stack([t,np.asarray(V),np.asarray(G)]).T,columns=['t','Vext','Gehp'])

    tVG.to_csv(tVG_name,sep=' ',index=False,float_format='%.3e')