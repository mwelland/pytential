from .. import pytential
from scipy.optimize import minimize
from scipy.interpolate import LinearNDInterpolator
from numpy import array, vstack
import numpy as np

class pyt_minimizer:
    def __init__(self, pyt, vars_out, constraints = None, default_initial_guess = None):
        #If the default initial guess is not specified, use .5 for all variables
        if default_initial_guess is None:
            default_initial_guess = .5*np.ones(len(pyt.vars))

        fcn_lmbda = lambda x: pyt.fcn(*x)

        #If there are no constraints specified, use all the constraints in pyt
        if constraints is None:
            pyt_constraints = [{'type': 'eq', 'fun': lambda x: f(*x)} for f in pyt.constraints]
        
        # Additional constraints to map variables to variables out.
        assert all(v in pyt.vars for v in vars_out), "All variables_out must be in pyt.vars"
        m = array([[1 if vo == vi else 0 for vi in vars_out] for vo in pyt.vars])

        self.fcn = fcn_lmbda
        self.pyt_const = pyt_constraints
        self.m = m
        self.x0_default = default_initial_guess

    def __call__(self, y, x0=None):
        if x0 is None: 
            x0 = self.x0_default

        ans = minimize(self.fcn, x0,
            constraints= self.pyt_const +
            [{'type': 'eq', 'fun': lambda x: self.m.T.dot(x) - y}],
            method="SLSQP")
        assert ans.success, "Minimization failed at y = " + str(y)
        return ans
        


class min_pytential(pytential):
    """
    Creates a pytential that runs a minimizer for every input 

    Reduces the dimensionality of the pytential by eliminating constraints and associated variables. 

    # Attributes:
    #     vars: an ordered list of the variable names as strings
    #     constraints: if left blank, all the constraints in pytential will be used.

    """
    
    def __init__(self, pyt, vars_out, constraints = None, default_initial_guess = None):

        assert isinstance(pyt, pytential), "Takes a pytential as an arguement"

        min = pyt_minimizer(pyt, vars_out, constraints = constraints, default_initial_guess = default_initial_guess)

        #TODO: #8 This isn't implemented well. I should return the full derivative_structure from the minimizer, filling in where necessary. Also include the minimum values to be passed to the next y.
        fcn = lambda y: min(y).fun
        super().__init__(fcn, vars_out)
        
        #def find_min(y, x0=None):
            #if x0 is None:
            #    x0 = np.ones(len(variables))
            # min = minimize_function(y, x0)
            # def min(y, x0):
            #     return minimize(fcn_lmbda, x0,
            #                 constraints= pyt_constraints +
            #                 [{'type': 'eq', 'fun': lambda x: m.T.dot(x) - y}],
            #                 method="SLSQP")
            # for i in range(len(y)):
            #     #x0_i = LinearNDInterpolator(y[:i-1], x0[:i])
            #     #x0_i = x0[-1] 
            #     #print(x0[-1])
            #     result = min(y[i], x0)
            #     #x0 = vstack((x0, result.x))
            #     #print(result)
            #     #print(x0)
            #     # if not ans.success:
            #     #     print("Minimization failed at y = ", y[i])
            #     #     print(ans.message)
            # result = min(y, x0).fun      
            # return result
       
        return 