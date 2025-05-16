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

        # # Wrapper to handle broadcasting over rows of a 2D array
        # def minimize_callable(arr):
        #     """
        #     Broadcasts min_fcn and collects only the minimized funciton value. 
        #     """
        #     return np.array([self.min_fcn(arr[:, i])[0] for i in range(arr.shape[1])])


        super().__init__(lambda x: self.min_fcn(x)[0], free_vars)
        
        self.parent = objective_pyt


    def min_fcn(self, arr):
        """
        Minimization function for a single set of free arguments.

        Args:
            free_args (array-like): Values for the free variables.
            objective_pyt (pytential): The objective pytential to minimize.
            free_vars (list): List of free variable names.

        Returns:
            dict: The full results dictionary from minimize_pytential.
        """
        def minimize_single(free_args):
            restriction_vars_dict = dict(zip(self.vars, free_args))
            pyt_reduced = self.parent.set_variables(restriction_vars_dict)
            res = minimize_pytential(pyt_reduced)
        
            optimized_vars_dict = dict(zip(pyt_reduced.vars, res.x))
            vars_dict = {**restriction_vars_dict, **optimized_vars_dict}

            return res.fun, vars_dict
        res = [minimize_single(arr[:, i]) for i in range(arr.shape[1])]
        values, vars_dicts = zip(*res)
        values = np.array(values)
        vars_dicts = list(vars_dicts)
        return values, vars_dicts



