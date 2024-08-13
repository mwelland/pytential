from .. import pytential
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds
from scipy.interpolate import LinearNDInterpolator
from numpy import array, vstack, vectorize
import numpy as np

class pyt_minimizer:
    def __init__(self, pyt, vars_out, constraints = None, x0_default = None):
        #If the default initial guess is not specified, use .5 for all variables
        n = len(pyt.vars)
        if x0_default is None:
            x0_default = .5*np.ones(n)

        #The minimizer expects a vector of varables so we define helper functions for the objective and constraints. 
        #IDEA: maybe pytential functions can accept a vector directly and distribute according to vars?
        def vec_args(func): return lambda x: func(*x)

        #If there are no constraints specified, use all the constraints in pyt
        def zero_hess(x,v): return np.zeros((n, n))
        pyt_constraints = [NonlinearConstraint(vec_args(f), 0, 0, hess = zero_hess) for f in pyt.constraints]

        # Additional constraints to map variables to variables out.
        assert all(v in pyt.vars for v in vars_out), "All variables_out must be in pyt.vars"
        m = array([[1 if vo == vi else 0 for vi in pyt.vars] for vo in vars_out])

        self.fcn = vec_args(pyt.fcn)
        self.jac = vec_args(pyt.grad)
        self.hess = vec_args(pyt.hess) 
        self.bounds = Bounds(0, np.inf)
        self.pyt_constraints = pyt_constraints
        self.vars_out_constraints = lambda y: LinearConstraint(m, y, y);     
        self.x0_default = x0_default
        
    def min(self, y, x0):
        # Put linear constraints as nonlinear so I can add the parameter instead of recreating each time?

        ans = minimize(self.fcn, 
            x0,
            jac = self.jac,
            hess = self.hess,
            bounds = self.bounds,
            constraints= self.pyt_constraints +
            [self.vars_out_constraints(y)],
            method="trust-constr",  
            options={"factorization_method":"SVDFactorization"},
            )
        
        return ans

    def find_min(self, y, x0=None):
        #TODO: #10 Insert a way to propogate the minimizer to the next run through interpolation
        if x0 is None: 
                x0 = self.x0_default
        ans = self.min(y,x0)
        assert ans.success, "Minimization failed at y = " + str(y)
        return ans.fun #vectorize(min, excluded='x0')(y,x0)[0]

    def guess_x0(y, x0=None):
        #Takes a (historical) list of y vs x0 and interpolates to guess the next x0. 
        pass
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
    
    def __init__(self, pyt, vars_out, constraints = None, x0_default = None):

        assert isinstance(pyt, pytential), "Takes a pytential as an arguement"
        
        min = pyt_minimizer(pyt, vars_out, constraints = constraints, x0_default = x0_default)

        #TODO: #8 This isn't implemented well. I should return the full derivative_structure from the minimizer, filling in where necessary. Also include the minimum values to be passed to the next y.
        # Working but needs revision. Issue with passing the min funciton into the eval / exec which would be cleaner.
        lambda_args = ", ".join(vars_out)
        lambda_body = f"lambda func: lambda {lambda_args}: func([{lambda_args}])"
        f = eval(lambda_body)
        fcn = vectorize(f(min.find_min))

        super().__init__(fcn, vars_out)
        self.minimized_vars = pyt.vars
       
        return 