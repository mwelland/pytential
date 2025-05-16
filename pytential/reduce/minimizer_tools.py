from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds
from numpy import inf
from .matrix_methods import get_constraints_matrix_and_vector
import numpy as np
import warnings

def minimize_pytential(pyt):
    # Minimizes the pytential over all free variables subject to the constraints
    nlc = [NonlinearConstraint(c, 0, 0) for c in pyt.constraints]
    lc = ({'type': 'eq', 'fun': c} for c in pyt.constraints)
    
    n = len(pyt.vars)
    def zero_hess(x,v): return np.zeros((n, n))
    
    pyt_constraints = [NonlinearConstraint(c, 0, 0, hess = zero_hess) 
                        for c in pyt.constraints]
    
    A,b = get_constraints_matrix_and_vector(pyt.constraints, len(pyt.vars))
    x0 = np.linalg.lstsq(A, -b, rcond=None)[0].flatten()

    bounds = Bounds([1e-6]*n, [inf]*n, keep_feasible=True)
    x0 = np.clip(x0, bounds.lb, bounds.ub)
    # TODO: #14 GEt a global minimizer working here? Issues with constraints...?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            pyt._fcn,
            x0,
            method='SLSQP',
            # method='trust-constr',
            jac=pyt._grad,
            #hess=pyt._hess,
            bounds=bounds,
            constraints=lc
        )


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