from .. import pytential
from .minimizer_tools import minimize_pytential

import numpy as np

class min_pytential(pytential):
    """
    Creates a pytential that runs a minimizer for every input 

    Reduces the dimensionality of the pytential by eliminating constraints and associated variables. 

    # Attributes:
    #     vars: an ordered list of the variable names as strings
    #     constraints: if left blank, all the constraints in pytential will be used.

    """
    
    def __init__(self, objective_pyt, free_vars):

        

        assert isinstance(objective_pyt, pytential), "Objective pytential must be a pytential"
        
        """
        Creates a partial function for f_constrained.fcn with some variables fixed.

        Args:
            fcn (callable): The original function (e.g., f_constrained.fcn).
            vars (list): The list of all variable names.
            fixed_vars (dict): A dictionary of fixed variable values (e.g., {'x': 0.5}).

        Returns:
            callable: A function that takes only the unfixed variables as input.
        """

        # Wrapper to handle broadcasting over rows of a 2D array
        def min_fcn_broadcast(free_args_array):
            """
            Broadcasted version of min_fcn to handle array inputs.

            Args:
                free_args_array: 2D array where each row corresponds to a set of free variables.

            Returns:
                Array of minimized function values for each row of free_args_array.
            """
            #TODO: #13 This isn't completely flexible for ca= 2, cb = [1,2]
            free_args_array = np.array(free_args_array)
            if free_args_array.shape[0] == 1:
                return self.min_fcn(free_args_array[0])[0]
            else: 
                return np.apply_along_axis(lambda args: self.min_fcn(args)[0], axis=0, arr=free_args_array)


        super().__init__(lambda x: min_fcn_broadcast(x), free_vars)
        
        self.parent = objective_pyt
        #self.min_fcn = min_fcn


    # TODO: #15 Fix the broadcasting issue here
    def min_fcn(self, free_args):
        """
        Minimization function for a single set of free arguments.

        Args:
            free_args (array-like): Values for the free variables.
            objective_pyt (pytential): The objective pytential to minimize.
            free_vars (list): List of free variable names.

        Returns:
            dict: The full results dictionary from minimize_pytential.
        """
        restriction_vars_dict = dict(zip(self.vars, free_args))
        pyt_reduced = self.parent.set_variables(restriction_vars_dict)
        res = minimize_pytential(pyt_reduced)
        optimized_vars_dict = dict(zip(pyt_reduced.vars, res.x))

        vars_dict = {**restriction_vars_dict, **optimized_vars_dict}

        return res.fun, vars_dict

       
        return 
