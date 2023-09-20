## Fitparam class
# Version 0.1
# (c) Larry Lueer, Vincent M. Le Corre, i-MEET 2021-2022

# Importing libraries
import numpy as np

class Fitparam():
    def __init__(self, name = '', startVal = None, val = 1.0, relRange = 0, lims = [], std = 0, d = '',display_name='',unit='',
                range_type = 'log', lim_type = 'absolute', optim_type = 'linear',axis_type = None, val_type = 'float',rescale = True,stepsize = None):
        """ Fitparam class object

        Parameters
        ----------
        name : str, optional
            name by which object can be retrived, by default ''
        startVal : float, optional
            starting guess for the optimization, by default None
        val : float, optional
            achieved value after the optimizaton, by default 1.0
        relRange : float, optional
            allowed variation range. Interpretation depends on range_type\\
            if relRange=0, parameter is considered fixed, by default 0
        lims : list, optional
            hard limits, by default []
        std : float, optional
            standard deviation as returened from optimization, by default 0
        d : str, optional
            parameter description, by default ''
        display_name : str, optional
            name to be displayed in plots, by default ''
        unit : str, optional
            unit of the parameter, by default ''
        range_type : str, optional
            Interpretation of relRange, by default 'log'
        lim_type : str, optional
            Respect hard limits set in lims ('absolute')\\
            or relative limits controlled by relRange('relative'), by default 'absolute'
        optim_type : str, optional
            Interpretation by optimizer ('linear' or 'log'), by default 'linear'
        axis_type : str, optional
            Set the type of scale and formating for the axis of the plots ('linear' or 'log') if let to None we use optim_type, by default None
        val_type : str, optional
            type of the parameter, can be 'float' or 'int' or 'str', by default 'float'
        rescale : bool, optional
            rescale the parameter to the order of magnitude of the startVal, by default True
        stepsize : float, optional
            stepsize for integer parameters (val_type = 'int'), by default None

        Raises
        ------
        ValueError
            range_type must be 'linear', 'lin', 'logarithmic' of 'log'
        ValueError
            lim_type must be 'absolute' or 'relative'
        ValueError
            optim_type must be 'linear', 'lin', 'logarithmic' of 'log'
        ValueError
            axis_type must be 'linear', 'lin', 'logarithmic' of 'log'

       
        """        

        self.name = name
        if display_name == '':
            self.display_name = name
        else:
            self.display_name = display_name

        self.unit = unit

        if unit == '':
            self.full_name = self.display_name
        else:
            self.full_name = self.display_name + ' [' + self.unit + ']'
        
        self.range_type = range_type
        self.lim_type = lim_type
        self.optim_type = optim_type
        self.axis_type = axis_type
        if self.axis_type == None:
            self.axis_type = self.optim_type
        
        self.val_type = val_type
        self.stepsize = stepsize


        # Check if limits are valid
        if self.range_type not in ['linear', 'log','lin','logarithmic']:
            raise ValueError('range_type must be ''linear'', ''lin'', ''logarithmic'' of ''log''')

        if self.lim_type not in ['absolute', 'relative']:
            raise ValueError('lim_type must be ''absolute'' or ''relative''')
        
        if self.optim_type not in ['linear', 'log','lin','logarithmic']:
            raise ValueError('optim_type must be ''linear'', ''lin'', ''logarithmic'' of ''log''')
        
        if self.axis_type not in ['linear', 'log','lin','logarithmic']:
            raise ValueError('axis_type must be ''linear'', ''lin'', ''logarithmic'' of ''log''')

        if self.val_type not in ['float', 'int','str']:
            raise ValueError('val_type must be ''float'', ''int'', ''str''')

        if self.range_type == 'lin': # correct for improper range_type
            self.range_type = 'linear'
        elif self.range_type == 'logarithmic':
            self.range_type = 'log'
        
        if self.optim_type == 'lin': # correct for improper optim_type
            self.optim_type = 'linear'
        elif self.optim_type == 'logarithmic':
            self.optim_type = 'log'
        
        if self.axis_type == 'lin': # correct for improper axis_type
            self.axis_type = 'linear'
        elif self.axis_type == 'logarithmic':
            self.axis_type = 'log'


        if self.val_type != 'str':
            if startVal == None:
                self.startVal = val *1.0
            else:
                self.startVal = startVal
            self.val = val *1.0
        else:
            if startVal == None:
                self.startVal = val
            else:
                self.startVal = startVal
            self.val = val


       
        self.relRange = relRange
        
        if lims ==[] and self.val_type == 'float':
            if self.relRange == 0:
                rr = 0.001
            else:
                rr = self.relRange

            if self.optim_type == 'linear':
                self.lims = [self.startVal  - rr*abs(self.startVal),self.startVal  + rr*abs(self.startVal)]
            elif self.optim_type == 'log':
                self.lims = [self.startVal *10**(-rr),self.startVal *10**(rr)]
            else:
                raise ValueError('optim_type must be ''linear'' or ''log''')

            # self.lims = [self.val - rr*abs(self.val),self.val + rr*abs(self.val)]
        else:
            self.lims = lims
        
        self.std = std
        self.d = d
        self.rescale = rescale

        
        

    def __str__(self):
        """ String representation of the FOMparam object with all attributes when printed
        Returns
        -------
        str
            string representation of the FOMparam object with all attributes
        """
        all_attributes = vars(self)
        attribute_string = "\n".join(f"{key}: {value}" for key, value in all_attributes.items() if not key.startswith("__"))
        return attribute_string
    
    def __repr__(self):
        """ String representation of the FOMparam object

        Returns
        -------
        str
            string representation of the FOMparam object
        """   
        return f"{self.__class__.__name__}(name={self.name}, val={self.val}, relRange={self.relRange}, lims={self.lims}, std={self.std}, d={self.d}, display_name={self.display_name}, unit={self.unit}, range_type={self.range_type}, lim_type={self.lim_type}, optim_type={self.optim_type}, axis_type={self.axis_type})"

    

class FOMparam():
    def __init__(self, func, name = '',  val = 1.0, std = 0, relRange = 1, display_name='', unit ='', optim_type = 'linear',axis_type = None):
        """ FOMparam class object

        Parameters
        ----------
        func : function
            function to calculate the FOMs
        name : str, optional
            name by which object can be retrived, by default ''
        val : float, optional
            achieved value after the optimizaton, by default 1.0
        relRange : float, optional
            allowed variation range. Interpretation depends on range_type\\
            if relRange=0, parameter is considered fixed, by default 0
        lims : list, optional
            hard limits, by default []
        std : float, optional
            standard deviation as returened from optimization, by default 0
        d : str, optional
            parameter description, by default ''
        display_name : str, optional
            name to be displayed in plots, by default ''
        unit : str, optional
            unit of the parameter, by default ''    
        range_type : str, optional
            Interpretation of relRange, by default 'log'
        lim_type : str, optional
            Respect hard limits set in lims ('absolute')\\
            or relative limits controlled by relRange('relative'), by default 'absolute'
        optim_type : str, optional
            Interpretation by optimizer ('linear' or 'log'), by default 'linear'
        axis_type : str, optional
            Set the type of scale and formating for the axis of the plots ('linear' or 'log') if let to None we use optim_type, by default None

        Raises
        ------
        ValueError
            range_type must be 'linear', 'lin', 'logarithmic' of 'log'
        ValueError
            lim_type must be 'absolute' or 'relative'
        ValueError
            optim_type must be 'linear', 'lin', 'logarithmic' of 'log'
        ValueError
            axis_type must be 'linear', 'lin', 'logarithmic' of 'log'


        """        

        self.name = name

        self.relRange = relRange
        if display_name == '':
            self.display_name = name
        else:
            self.display_name = display_name

        self.unit = unit
        if unit == '':
            self.full_name = self.display_name
        else:
            self.full_name = self.display_name + ' [' + self.unit + ']'

        self.func = func

        self.val = val *1.0
       
        self.std = std

        self.optim_type = optim_type

        if optim_type not in ['linear', 'log','lin','logarithmic']:
            raise ValueError('optim_type must be ''linear'', ''lin'', ''logarithmic'' of ''log''')

        if optim_type == 'lin': # correct for improper optim_type
            self.optim_type = 'linear'
        elif optim_type == 'logarithmic':
            self.optim_type = 'log'

        self.axis_type = axis_type
        if self.axis_type == None:
            self.axis_type = self.optim_type
    
    def update_FOMparam(self,FOM_list):
        """ Update the FOM_param object with lims and startVal based on the FOM_list
        set the lims to the min and max of the FOM_list and the startVal to the mean
        set the relRange to 1
        set the lim_type to absolute

        Parameters
        ----------
        FOM_list : list
            list of FOMs
        """        
        # for i in range(len(FOMs)): # update FOM_param objects
        if self.optim_type == 'log':
            self.lims = [10**(min(FOM_list)),10**(max(FOM_list))]
            self.startVal = (self.lims[0]+self.lims[1])/2
        else:
            self.lims = [min(FOM_list),max(FOM_list)]
            self.startVal = (max(FOM_list)+min(FOM_list))/2
            if min(FOM_list) != 0:
                self.p0m = 10**(np.floor(np.log10(np.abs(min(FOM_list))))) # the order of magnitude of the parameters
            elif max(FOM_list) != 0:
                self.p0m = 10**(np.floor(np.log10(np.abs(max(FOM_list)))))
            else:
                self.p0m = 1
        # self.startVal = (max(FOM_list)+min(FOM_list))/2
        self.lim_type = 'absolute'
        self.relRange = 1
    
        
        

    def __str__(self):
        """ String representation of the FOMparam object with all attributes when printed
        Returns
        -------
        str
            string representation of the FOMparam object with all attributes
        """        
        all_attributes = vars(self)
        attribute_string = "\n".join(f"{key}: {value}" for key, value in all_attributes.items() if not key.startswith("__"))
        return attribute_string
    
    def __repr__(self):
        """ String representation of the FOMparam object

        Returns
        -------
        str
            string representation of the FOMparam object
        """        
        return f"{self.__class__.__name__}(name={self.name}, val={self.val}, relRange={self.relRange}, lims={self.lims}, std={self.std}, func={self.func}, display_name={self.display_name}, unit={self.unit}, optim_type={self.optim_type}, axis_type={self.axis_type})"