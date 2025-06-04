from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds, shgo
from numpy import inf
#from ..quadratic_pytential.matrix_methods import get_constraints_matrix_and_vector
import numpy as np
import warnings

def minimize_pytential(pyt):
    # Minimizes the pytential over all free variables subject to the constraints
    # nlc = [NonlinearConstraint(c, 0, 0) for c in pyt.constraints]
    lc = ({'type': 'eq', 'fun': c} for c in pyt.constraints)
    
    n = len(pyt.vars)
    def zero_hess(x,v): return np.zeros((n, n))
    
    # pyt_constraints = [NonlinearConstraint(c, 0, 0, hess = zero_hess) 
    #                     for c in pyt.constraints]
    
    A,b = pyt.get_constraint_matrix_and_vector()# get_constraints_matrix_and_vector(pyt.constraints, len(pyt.vars))
    
    b = np.atleast_1d(b).flatten()  # Ensure b is 1D
    bounds = Bounds([1e-6]*n, [inf]*n, keep_feasible=True)
    
    if A.size > 0:
        linear_constraint = LinearConstraint(A, -b, -b)
        x0 = np.linalg.lstsq(A, -b, rcond=None)[0].flatten()
        x0 = np.clip(x0, bounds.lb, bounds.ub)
    else:
        linear_constraint = None
        x0 = np.array([.5]*n)  # Default starting point if no constraints
      
    


    

    def safe_objective(x):
        """Objective function wrapper to handle log domain errors."""
        if np.all(x > 0):
            return pyt._fcn(x)
        else:
            return np.inf
        
    
    def safe_gradient(x):
        if np.all(x > 0):
            return pyt._grad(x)
        else:
            return np.inf*x
        
        
    # def safe_hessian(x):
    #     return pyt._hess(x)
       
    # TODO: #14 Get a global minimizer working here? Issues with constraints...?

    # res = minimize(
    #         safe_objective,
    #         x0,
    #         method='trust-constr',
    #         jac=safe_gradient,
    #         hess=safe_hessian,
    #         bounds=bounds,
    #         constraints=linear_constraint, #pyt_constraints,
    #         #options={'trust_region_tol': 1e-8}  # Adjust trust_region_tol here
    #     )


    # res = shgo(
    #     func=safe_objective,
    #     #jac=safe_gradient, # Pass the Jacobian function here
    #     bounds=bounds,
    #     constraints=lc,
    #     #options={'disp': False} # Set to True for more output


    
    res = minimize(
            pyt._fcn,
            x0,
            method='SLSQP',
            jac=pyt._grad,
            bounds=bounds,
            constraints=lc
    )



    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     res = minimize(
    #         pyt._fcn,
    #         x0,
    #         method='SLSQP',
    #         jac=pyt._grad,
    #         hess=pyt._hess,
    #         bounds=bounds,
    #         constraints=lc
    #     )
    # with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
        


    # bnds = Bounds(lb, ub, keep_feasible=True)
    # res = minimize(pyt._fcn,
    #         x0,
    #         method='trust-constr',
    #         jac=grad,
    #         hess=hess,
    #         bounds=bnds,
    #         constraints=constraints,
    #         options={'verbose': 2})

    return res