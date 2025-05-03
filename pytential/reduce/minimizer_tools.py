from scipy.optimize import minimize, NonlinearConstraint, shgo, LinearConstraint, Bounds, differential_evolution
from numpy import inf
from .matrix_methods import get_constraints_matrix_and_vector
import numpy as np

def minimize_pytential(pyt):
    # Minimizes the pytential over all free variables subject to the constraints
    nlc = [NonlinearConstraint(c, 0, 0) for c in pyt.constraints]
    lc = ({'type': 'eq', 'fun': c} for c in pyt.constraints)
    bounds = [(0.001, .999)] * len(pyt.vars)
    # res = differential_evolution(
    #     pyt.fcn, bounds,
    #     constraints=(nlc,),
    #     updating="deferred",
    #     maxiter=500,
    #     tol=1e-6,
    #     polish=True
    # )
    n = len(pyt.vars)
    def zero_hess(x,v): return np.zeros((n, n))
    # minimizer_kwargs = {
    # 'method': 'SLSQP',
    # 'constraints': lc,        # your list/dict of constraints
    # }
    # res = shgo(fun, bounds, constraints=cons,
    #        minimizer_kwargs=minimizer_kwargs)
    
    pyt_constraints = [NonlinearConstraint(c, 0, 0, hess = zero_hess) 
                        for c in pyt.constraints]
    # res = shgo(
    #     pyt.fcn, 
    #     bounds,
    #     constraints=pyt.constraints,
    #     minimizer_kwargs=minimizer_kwargs,
    #     options={'disp': True}
    # )

    A,b = get_constraints_matrix_and_vector(pyt.constraints, len(pyt.vars))
    x0 = np.linalg.lstsq(A,-b)[0]
    print(x0)


    res = minimize(pyt._fcn, 
        x0,
        jac = pyt._grad,
        hess = pyt._hess,
        bounds = Bounds(0, inf),
        constraints= pyt_constraints,
        method="trust-constr",  
        options={"factorization_method":"SVDFactorization"},
        )
    return res