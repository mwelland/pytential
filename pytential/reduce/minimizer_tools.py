from scipy.optimize import minimize, NonlinearConstraint, shgo, LinearConstraint, Bounds

def minimize_pytential(pyt):
    # Minimizes the pytential over all free variables subject to the constraints
    nlc = [NonlinearConstraint(c, 0, 0) for c in pyt.constraints]
    bounds = [(0.001, .999)] * len(pyt.vars)
    # res = differential_evolution(
    #     obj, bounds,
    #     constraints=(nlc,),
    #     updating="deferred",
    #     maxiter=500,
    #     tol=1e-6,
    #     polish=True
    # )
    res = shgo(
        pyt.fcn, 
        bounds,
        constraints=nlc,
        sampling_method='simplicial',
        #options={'maxiter':100}
    )
    return res