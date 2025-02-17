from sympy import Matrix, symbols, ln, ones, oo

def function_from_properties(properties):
    #Returns a sympy function consistent with the properties dictionary
    
    assert type(properties) is dict, "properties must be a dictionary"

    T, V = symbols('T V')


    # Ensure properties['mu0'] is set and is a Matrix
    if 'mu0' not in properties or not properties['mu0']:
        raise ValueError("properties['mu0'] must be set and non-empty")
    elif not isinstance(properties['mu0'], Matrix):
        properties['mu0'] = Matrix(properties['mu0'])

    n = len(properties['mu0'])
    keys = properties.keys()
    
    # if not 'rho' in keys:
    #     print('Density, rho, not found. Assuming rho = 1')
    #     properties['rho'] = 1

    # if not 'vi' in keys:
    #     print('Specific volumes, vi, not found. Assuming vi = 1/rho')
    #     properties['vi'] = [1/properties['rho']]*n
    # elif len(properties['vi']) == 1:
    #     print('Only one specific volume entered. Propogating to all')
    #     properties['vi'] = [properties['vi']]*n
    
    fcn = 0
    constraints = []
    
    
    ns = Matrix(symbols('n:{}'.format(n)))



    fcn += _ideal_mixing(ns, properties['mu0'], T = T)

    # If 'kappa' is defined, assume hyperelastic. Else, lattice constraints.
    if 'kappa' in keys:
        if 'vi' in keys:
            vi = properties['vi']
        else:
            vi = ones(n,1)
        
        if properties['kappa'] == oo:
            constraints += [_lattice_constraint(ns, properties['mu0'])]
        else:
            fcn += _strain_energy(ns, vi, V, properties['kappa'])
            
    #TODO: #5 option in sympyt funciton to not include lattice constraint
    #TODO #7 Output directly usable in sympy_pytential
    return fcn, constraints





def _ideal_mixing(ns, mu0, T = 300):
    """
    Calculates the ideal mixing terms. Degenerates to the 1-component case if len(mu0) == 1.

    Args:
        ns (Matrix): The matrix of abundances.
        mu0 (Matrix): The matrix of reference chemical potentials.
        T (symbol, default = 300): The temperature. Defaults to 300.

    Returns:
        Expr: The sympy expression for the ideal mixing terms.
    """
    xs = ns/sum(ns)
    RT = 8.314*T
    return ns.dot(mu0 + RT*xs.applyfunc(ln))

def _lattice_constraint(ns, vs, V = 1):
    """
    Returns the lattice constraint expression.

    Args:
        ns (Matrix): The matrix of abundances.
        vs (Matrix): The matrix of specific volumes.
        V (symbol, default = 1): The volume.

    Returns:
        Expr: The sympy expression for the lattice constraint.
    """
    return ns.dot(vs)-V

def _strain_energy(ns, vs, V, kappa, method = 'NeoHookean'):
    """
    Returns the strain energy expression.

    Args:
        ns (Matrix): The matrix of abundances.
        vs (Matrix): The matrix of specific volumes.
        V (symbol, default = 1): The volume.
        kappa (Expr): The bulk modulus.
        method (str, default = 'NeoHookean'): The method to use.

    Returns:
        Expr: The sympy expression for the lattice constraint.
    """
    V0 = ns.dot(vs)
    delta = V/V0
    if method == "NeoHookean":
        f_el = V0*kappa/2*(ln(delta))**2
    return f_el
