from .. import pytential
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds
from scipy.interpolate import NearestNDInterpolator
from numpy import array, vstack, vectorize
import numpy as np

class pyt_minimizer:
    """
    An object that finds the minimum of a pytential function, subject to its constraints
    for a given set of output variables

    Bounds and constraint objects are setup at initialization and persist over runs

    Keeps a record of successful minimization runs to provide intelligent initial guesses through interpolation
    """
    def __init__(self, objective_pyt, vars_out):
        #If the default initial guess is not specified, use .5 for all variables
        
        self.objective_pyt = objective_pyt

        #The minimizer expects a vector of varables so we define helper functions for the objective and constraints. 
        #IDEA: maybe pytential functions can accept a vector directly and distribute according to vars?
        #def vec_args(func): return lambda x: func(*x)

        #Use all the constraints in pyt. Change in future?
        n = len(objective_pyt.vars)
        def zero_hess(x,v): return np.zeros((n, n))
        pyt_constraints = [NonlinearConstraint(c, 0, 0, hess = zero_hess) 
                           for c in objective_pyt.constraints]

        # Additional constraints to map variables to variables out.
        assert all(v in objective_pyt.vars for v in vars_out), "All variables_out must be in pyt.vars"
        m = array([[1 if vo == vi else 0 for vi in objective_pyt.vars] for vo in vars_out])

        self.vars_out = vars_out
        self.bounds = Bounds(0, np.inf)
        self.pyt_constraints = pyt_constraints
        self.vars_out_constraints = lambda y: LinearConstraint(m, y, y);
        self.x_vs_y = []
        
    def min(self, y, x0):
        # Performs the minimization for a single value of y, given x0. 
        # Returns the results object. 

        # NOTE: Is it worthwhile making the linear constraints nonlinear?
        pyt = self.objective_pyt
        ans = minimize(pyt._fcn, 
            x0,
            jac = pyt._grad,
            hess = pyt._hess,
            bounds = self.bounds,
            constraints= self.pyt_constraints +
            [self.vars_out_constraints(y)],
            method="trust-constr",  
            options={"factorization_method":"SVDFactorization"},
            )
        return ans

    def find_single_min(self, y):
        # Finds the minimum for a single y, processes results

        x0 = self.predict_x0(y)
        print('xo used', x0)
        result = self.min(y, x0)
        assert result.success, f"Minimization failed at y = {y}"
        self.x_vs_y.append([result.x, y])
        return result

    def find_min(self, y):
        # Finds the minimum for any number of y, processes results
        return [self.find_single_min(yi) for yi in y]

        #TODO: #10 Insert a way to propogate the minimizer to the next run through interpolation

    def predict_x0(self, y):
        if self.x_vs_y == []:
            x0 = .5*np.ones( len(self.objective_pyt.vars) )
        else:
            #print('x vs y', self.x_vs_y)
            x_history, y_history = zip(*[(xy[0], xy[1]) for xy in self.x_vs_y])       
            x_history = np.array(x_history)
            y_history = np.array(y_history)

            x0 = NearestNDInterpolator(y_history, x_history)(y)[0]
            #print('predicted_x0', x0)

            #x0 = .5*np.ones( len(self.objective_pyt.vars) )


        return x0


    def __call__(self, y, x0=None):
        #[self.find_min(yi, x0) for yi in y]
        return self.find_min(y, x0)
        


class min_pytential(pytential):
    """
    Creates a pytential that runs a minimizer for every input 

    Reduces the dimensionality of the pytential by eliminating constraints and associated variables. 

    # Attributes:
    #     vars: an ordered list of the variable names as strings
    #     constraints: if left blank, all the constraints in pytential will be used.

    """
    
    def __init__(self, objective_pyt, vars_out):

        assert isinstance(objective_pyt, pytential), "Takes a pytential as an arguement"

        min = pyt_minimizer(objective_pyt, vars_out)
       
        #fcn = min.find_min
        fcn = min.find_min
        

        #Vectorize isn't working right. Somehow it is not taking a list of arrays but rather each element of the inputs. 

        super().__init__(fcn, vars_out)
        self.minimized_vars = objective_pyt.vars
       
        return 
