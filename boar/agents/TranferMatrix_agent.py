import os, sys
import numpy as np
from scipy import constants
from scipy.interpolate import interp1d
from math import ceil
from copy import deepcopy
# BOAR imports
from boar.agents.Agent import Agent
from boar.core.funcs import callable_name, get_unique_X, get_unique_X_and_xaxis_values

# physical constants
q = constants.value(u'elementary charge')
c = constants.value(u'speed of light in vacuum')
h = constants.value(u'Planck constant')

class Transfer_Matrix_agent(Agent):
    """ Agent to run drift diffusion simulations with SIMsalabim to be used with BOAR MultiObjectiveOptimizer
    
    Parameters
    ----------
    

    """
    def __init__(self,layers=None,thicknesses=None,activeLayer=None,lambda_start=None,lambda_stop=None,lambda_step=None,x_step=1, mat_dir='Example_Data/matdata',am15_file='AM15G.csv',unit_th='nm') -> None:
        super().__init__()

        self.layers = layers
        self.thicknesses = thicknesses
        self.activeLayer = activeLayer
        self.lambda_start = lambda_start
        self.lambda_stop = lambda_stop
        self.lambda_step = lambda_step
        self.x_step = x_step

        if self.layers is None:
            self.layers = ['SiO2' , 'ITO' , 'PEDOT' , 'P3HTPCBM_BHJ' , 'Ca' , 'Al']
        if self.thicknesses is None:
            self.thicknesses  = [0 , 110 , 35  , 220 , 7 , 200] #nm
        if self.activeLayer is None:
            self.activeLayer = 3
        else:
            self.activeLayer = int(activeLayer)
        if self.lambda_start is None:
            self.lambda_start = 350 #nm
        if self.lambda_stop is None:
            self.lambda_stop = 800
        if self.lambda_step is None:
            self.lambda_step = 1
        if self.x_step is None:
            self.x_step = 1 #nm
        
        if unit_th not in ['nm','um','m']:
            raise ValueError('unit must be either nm or um or m')
        else:
            self.unit_th = unit_th
        if unit_th == 'm':
            self.thicknesses = [i*1e9 for i in self.thicknesses]
        elif unit_th == 'um':
            self.thicknesses = [i*1e3 for i in self.thicknesses]
        
        if len(self.layers) != len(self.thicknesses):
            raise ValueError('layers and thicknesses must have the same length, if the values are not known, use 0 as a placeholder')   
        
        if self.activeLayer > len(self.layers):
            raise ValueError('activeLayer must be below the number of layers')
        
        # make mad_dir absolute
        self.mat_dir = os.path.abspath(mat_dir)
        # check is am15_file is a string or a path
        if os.path.isfile(am15_file):
            self.am15_file = am15_file
        else:
            self.am15_file = os.path.join(self.mat_dir,am15_file)

    def openFile(self,fname):
        """ opens files and returns a list split at each new line

        Parameters
        ----------
        fname : string
            path to the file

        Returns
        -------
        fd : list
            list of lines in the file

        Raises
        ------
        ValueError
            Target is not a readable file

        """    
        fd = []
        if os.path.isfile(fname):
            fn = open(fname, 'r')
            fdtmp = fn.read()
            fdtmp = fdtmp.split('\n')
            # clean up line endings
            for f in fdtmp:
                f = f.strip('\n')
                f = f.strip('\r')
                fd.append(f)
            # make so doesn't return empty line at the end
            if len(fd[-1]) == 0:
                fd.pop(-1)
        else:
            print("%s Target is not a readable file" % fname)
        return fd

    def get_ntotal(self,matName,lambdas):
        """ get the complex refractive index of a material from a file

        Parameters
        ----------
        matName : string
            name of the material in the matdata folder
        lambdas : list
            list of wavelengths in nm

        Returns
        -------
        ntotal : list
            list of complex refractive index values
        """      
        matPrefix		= 'nk_'		    # materials data prefix  
        fname = os.path.join(self.mat_dir,'%s%s.csv' % (matPrefix,matName))
        matHeader = 0
        # check number of lines with strings in the header
        for line in self.openFile(fname):
            # check if line starts with a number
            if line[0].isdigit():
                break
            else:
                matHeader += 1
        
        fdata = self.openFile(fname)[matHeader:]
        # get data from the file
        lambList	= []
        nList		= []
        kList		= []
        for l in fdata:
            wl , n , k = l.split(',')
            wl , n , k = float(wl) , float(n) , float(k)
            lambList.append(wl)
            nList.append(n)
            kList.append(k)
        # make interpolation functions
        int_n	= interp1d(lambList,nList)
        int_k	= interp1d(lambList,kList)
        # interpolate data
        kintList	= int_k(lambdas)
        nintList	= int_n(lambdas)
        # make ntotal
        ntotal = []
        for i,n in enumerate(nintList):
            nt = complex(n,kintList[i])
            ntotal.append(nt)
        return ntotal

    def I_mat(self,n1,n2):
        """ calculate the interface matrix

        Parameters
        ----------
        n1 : float
            refractive index of the first material
        n2 : float
            refractive index of the second material

        Returns
        -------
        ret : array
            interface matrix
        """        
        r = (n1-n2)/(n1+n2)
        t = (2*n1)/(n1+n2)
        ret = np.array([[1,r],[r,1]],dtype=complex)
        ret = ret / t
        return ret

    def L_mat(self,n,d,l):
        """ calculate the propagation matrix

        Parameters
        ----------
        n : array
            complex refractive index of the material
        d : float
            thickness of the material
        l : float
            wavelength

        Returns
        -------
        L : array
            propagation matrix
        """        

        xi = (2*np.pi*d*n)/l
        L = np.array( [ [ np.exp(complex(0,-1.0*xi)),0] , [0,np.exp(complex(0,xi))] ] )
        return L



    def TM(self,X,params,output=['Jsc'],is_MOO=False):
        """ Calculate the Jsc, AVT or LUE for a multilayer stack

        Parameters
        ----------
        X : np.array
            Array of fixed parameters (not really relevant here)
        params : list
            list of Fitparam objects
        output : list, optional
            type of output, can be 'Jsc', 'AVT' or 'LUE', by default ['Jsc']
        is_MOO : bool, optional
            check if the function is called by a MOO algorithm, if True, the output is a dictionary with 'Jsc', 'AVT' and 'LUE' as keys, if not the output is a list with the values in the same order as output, by default False

        Returns
        -------
        dict or list
            dictionary or list with the values of the requested output

        Raises
        ------
        ValueError
            Wrong indices for the thicknesses
        ValueError
            Wrong indices for the complex refractive index
        """        
        
        # prepare the stack
        pnames = [p.name for p in params]

        # Read the parameters
        dnames = [p for p in pnames if p.startswith('d_')] #find parameters that start with 'd_'
        dindices = [int(p.split('_')[1]) for p in dnames] # read index after 'd_' for these parameters
        nknames = [p for p in pnames if p.startswith('nk_')]
        nkindices = [int(p.split('_')[1]) for p in nknames]

        # check that all indexes in dinices are in nkindices are below the number of layers
        maxindex = len(self.layers)
        if any([i>maxindex for i in dindices]):
            raise ValueError('dindices must be below the number of layers')
        if any([i>maxindex for i in nkindices]):
            raise ValueError('nkindices must be below the number of layers')

        t = deepcopy(self.thicknesses)
        lambdas	= np.arange(self.lambda_start,self.lambda_stop+self.lambda_step,self.lambda_step,float)
        layers = deepcopy(self.layers)
        x_step = deepcopy(self.x_step)
        activeLayer = deepcopy(self.activeLayer)

        # update the thicknesses
        for i in dindices:
            self.thicknesses[i] = [p.val for p in params if p.name == 'd_'+str(i)][0]
            t[i] = [p.val for p in params if p.name == 'd_'+str(i)][0]
        # update the nk values
        for i in nkindices:
            self.layers[i] = [p.val for p in params if p.name == 'nk_'+str(i)][0]
            layers[i] = [p.val for p in params if p.name == 'nk_'+str(i)][0]
        
        # load and interpolate AM1.5G Data
        am15_file = self.am15_file
        am15_data = self.openFile(am15_file)[1:]
        am15_xData = []
        am15_yData = []
        for l in am15_data:
            x,y = l.split(',')
            x,y = float(x),float(y)
            am15_xData.append(x)
            am15_yData.append(y)
        am15_interp = interp1d(am15_xData,am15_yData,'linear')
        am15_int_y  = am15_interp(lambdas)
        
        # load and interpolate human eye response
        photopic_file = os.path.join(self.mat_dir,'photopic_curve.csv')
        photopic_data = self.openFile(photopic_file)[0:]
        photopic_xData = []
        photopic_yData = []
        for l in photopic_data:
            x,y = l.split(',')
            x,y = float(x),float(y)
            photopic_xData.append(x)
            photopic_yData.append(y)
        photopic_interp = interp1d(photopic_xData,photopic_yData,'linear')
        photopic_int_y  = photopic_interp(lambdas)

        # ------ start actual calculation  --------------------------------------
        

        # initialize an array
        n = np.zeros((len(layers),len(lambdas)),dtype=complex)

        # load index of refraction for each material in the stack
        for i,l in enumerate(layers):
            ni = np.array(self.get_ntotal(l,lambdas))
            n[i,:] = ni

        # calculate incoherent power transmission through substrate

        T_glass = abs((4.0*1.0*n[0,:])/((1+n[0,:])**2))
        R_glass = abs((1-n[0,:])/(1+n[0,:]))**2

        # calculate transfer matrices, and field at each wavelength and position
        t[0] 		= 0
        t_cumsum	= np.cumsum(t)
        x_pos		= np.arange((x_step/2.0),sum(t),x_step)
        # get x_mat
        comp1	= np.kron(np.ones( (len(t),1) ),x_pos)
        comp2	= np.transpose(np.kron(np.ones( (len(x_pos),1) ),t_cumsum))
        x_mat 	= sum(comp1>comp2,0) 	# might need to get changed to better match python indices

        R		= lambdas*0.0
        T2		= lambdas*0.0
        E		= np.zeros( (len(x_pos),len(lambdas)),dtype=complex )

        # start looping
        for ind,l in enumerate(lambdas):
            # calculate the transfer matrices for incoherent reflection/transmission at the first interface
            S = self.I_mat(n[0,ind],n[1,ind])
            for matind in np.arange(1,len(t)-1):
                mL = self.L_mat( n[matind,ind] , t[matind] , lambdas[ind] )
                mI = self.I_mat( n[matind,ind] , n[matind+1,ind])
                S  = np.asarray(np.mat(S)*np.mat(mL)*np.mat(mI))
            R[ind] = abs(S[1,0]/S[0,0])**2
            T2[ind] = abs((2/(1+n[0,ind])))/np.sqrt(1-R_glass[ind]*R[ind])
            # this is not the transmittance! 
            # good up to here
            # calculate all other transfer matrices
            for material in np.arange(1,len(t)):
                    xi = 2*np.pi*n[material,ind]/lambdas[ind]
                    dj = t[material]
                    x_indices	= np.nonzero(x_mat == material)
                    x			= x_pos[x_indices]-t_cumsum[material-1]
                    # Calculate S_Prime
                    S_prime		= self.I_mat(n[0,ind],n[1,ind])
                    for matind in np.arange(2,material+1):
                        mL = self.L_mat( n[matind-1,ind],t[matind-1],lambdas[ind] )
                        mI = self.I_mat( n[matind-1,ind],n[matind,ind] )
                        S_prime  = np.asarray( np.mat(S_prime)*np.mat(mL)*np.mat(mI) )
                    # Calculate S_dprime (double prime)
                    S_dprime	= np.eye(2)
                    for matind in np.arange(material,len(t)-1):
                        mI	= self.I_mat(n[matind,ind],n[matind+1,ind])
                        mL	= self.L_mat(n[matind+1,ind],t[matind+1],lambdas[ind])
                        S_dprime = np.asarray( np.mat(S_dprime) * np.mat(mI) * np.mat(mL) )
                    # Normalized Electric Field Profile
                    num = T2[ind] * (S_dprime[0,0] * np.exp( complex(0,-1.0)*xi*(dj-x) ) + S_dprime[1,0]*np.exp(complex(0,1)*xi*(dj-x)))
                    den = S_prime[0,0]*S_dprime[0,0]*np.exp(complex(0,-1.0)*xi*dj) + S_prime[0,1]*S_dprime[1,0]*np.exp(complex(0,1)*xi*dj)
                    E[x_indices,ind] = num / den
        # overall Reflection from device with incoherent reflections at first interface
        Reflectance = R_glass+T_glass**2*R/(1-R_glass*R)

        # Absorption coefficient in 1/cm
        a = np.zeros( (len(t),len(lambdas)) )
        for matind in np.arange(1,len(t)):
            a[matind,:] = ( 4 * np.pi * np.imag(n[matind,:]) ) / ( lambdas * 1.0e-7 )

        # Absorption
        Absorption = np.zeros( (len(t),len(lambdas)) )
        for matind in np.arange(1,len(t)):
            Pos 		= np.nonzero(x_mat == matind)
            AbsRate 	= np.tile( (a[matind,:] * np.real(n[matind,:])),(len(Pos),1)) * (abs(E[Pos,:])**2)
            Absorption[matind,:] = np.sum(AbsRate,1)*x_step*1.0e-7

        # Transmittance
        Transmittance = 1 - Reflectance - np.sum(Absorption,0)
        Transmittance[Transmittance<0] = 0 # set negative values to zero

        # calculate generation profile
        ActivePos = np.nonzero(x_mat == activeLayer)
        tmp1	= (a[activeLayer,:]*np.real(n[activeLayer,:])*am15_int_y)
        Q	 	= np.tile(tmp1,(np.size(ActivePos),1))*(abs(E[ActivePos,:])**2)
        # Exciton generation rate
        Gxl		= (Q*1.0e-3)*np.tile( (lambdas*1.0e-9) , (np.size(ActivePos),1))/(h*c)
        if len(lambdas) == 1:
            lambda_step = 1
        else:
            lambda_step = (sorted(lambdas)[-1] - sorted(lambdas)[0])/(len(lambdas) - 1)
        Gx		= np.sum(Gxl,2)*lambda_step

        # calculate Jsc 
        Jsc = np.sum(Gx)*x_step*1.0e-7*q*1.0e3

        # calculate AVT and LUE
        AVT = sum(am15_int_y * photopic_int_y * Transmittance)/sum(am15_int_y * photopic_int_y)
        LUE = Jsc * AVT
        
        if is_MOO == False:
            if len(output) == 1:
                if output[0] == 'Jsc':
                    return Jsc
                elif output[0] == 'AVT':
                    return AVT
                elif output[0] == 'LUE':
                    return LUE
            else:
                res = []
                for out in output:
                    if out == 'Jsc':
                        res.append(Jsc)
                    elif out == 'AVT':
                        res.append(AVT)
                    elif out == 'LUE':
                        res.append(LUE)
                return res
        else:
            return {'Jsc':Jsc,'AVT':AVT,'LUE':abs(LUE)}

        