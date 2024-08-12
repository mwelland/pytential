from .. import pytential
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from scipy.interpolate import LinearNDInterpolator
from numpy import array, vstack, vectorize
import numpy as np

class pyt_minimizer:
    def __init__(self, pyt, vars_out, constraints = None, default_initial_guess = None):
        #If the default initial guess is not specified, use .5 for all variables
        if default_initial_guess is None:
            default_initial_guess = .5*np.ones(len(pyt.vars))

        #The minimizer expects a vector of varables so we define helper functions for the objective and constraints. 
        #IDEA: maybe pytential functions can accept a vector directly and distribute according to vars?

        fcn_lmbda = lambda x: pyt.fcn(*x)

        #If there are no constraints specified, use all the constraints in pyt
        if constraints is None:
            #pyt_constraints = [{'type': 'eq', 'fun': lambda x: f(*x)} for f in pyt.constraints]
            n = len(pyt.vars) #hess = lambda x: numpy.zeros((n, n))
            def cons_H(x,v): return v[0]*np.zeros((n, n))
            pyt_constraints = [NonlinearConstraint(lambda x, f=f:f(*x),0,0) for f in pyt.constraints]

        
        # Additional constraints to map variables to variables out.
        assert all(v in pyt.vars for v in vars_out), "All variables_out must be in pyt.vars"
        m = array([[1 if vo == vi else 0 for vi in pyt.vars] for vo in vars_out])
        
        self.fcn = fcn_lmbda
        self.pyt_const = pyt_constraints
        self.m = m
        self.x0_default = default_initial_guess

    def find_min(self, y, x0=None):
        
        
        def min(y, x0=None):
            if x0 is None: 
                x0 = self.x0_default
            
            # Inequality constraints: all variables must be positive
            ineq_constraints = NonlinearConstraint(lambda x:x, 0, np.inf)
            # Put linear constraints as nonlinear so I can add the parameter instead of recreating each time?

            ans = minimize(self.fcn, x0,
                constraints= self.pyt_const +
                [ineq_constraints] +
                [LinearConstraint(self.m, y, y)],
                method="trust-constr",
                )
            assert ans.success, "Minimization failed at y = " + str(y)
            
            return ans.fun
        
        #from numpy import vectorize
        return min(y,x0)#vectorize(min, excluded='x0')(y,x0)[0]

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
    
    def __init__(self, pyt, vars_out, constraints = None, default_initial_guess = None):

        assert isinstance(pyt, pytential), "Takes a pytential as an arguement"

        min = pyt_minimizer(pyt, vars_out, constraints = constraints, default_initial_guess = default_initial_guess)

        #TODO: #8 This isn't implemented well. I should return the full derivative_structure from the minimizer, filling in where necessary. Also include the minimum values to be passed to the next y.
  
        # Working but needs revision. Issue with passing the min funciton into the eval / exec which would be cleaner.
        lambda_args = ", ".join(vars_out)
        lambda_body = f"lambda func: lambda {lambda_args}: func([{lambda_args}])"
        f = eval(lambda_body)
        fcn = vectorize(f(min.find_min))

        super().__init__(fcn, vars_out)
        self.minimized_vars = pyt.vars
        
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