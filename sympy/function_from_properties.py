from sympy import Matrix, symbols, ln

def function_from_properties(properties, T = None):
    assert type(properties) is dict, "properties must be a dictionary"

    if T is None:
        T = symbols('T')

    n = len(properties['mu0'])
    keys = properties.keys()
    
    if not 'rho' in keys:
        print('Density, rho, not found. Assuming rho = 1')
        properties['rho'] = 1

    if not 'vi' in keys:
        print('Specific volumes, vi, not found. Assuming vi = 1/rho')
        properties['vi'] = [1/properties['rho']]*n
    elif len(properties['vi']) == 1:
        print('Only one specific volume entered. Propogating to all')
        properties['vi'] = [properties['vi']]*n
    
    fcn = 0
    constraints = None
    
    
    cs = Matrix(symbols('c:{}'.format(n)))

    fcn += ideal_mixing_term(cs, properties['mu0'], T = T, rho = properties['rho'])

    # If 'kappa' is defined, assume hyperelastic. Else, lattice constraints.
    if 'kappa' in keys:
        V = symbols('V')  
        fcn += hyperelastic_term(
            kappa = properties['kappa'], 
            vi = properties['vi'], 
            cs = cs, 
            V = V)
    else:
        constraints = cs.dot(properties['vi'])-properties['rho']
    #TODO: #5 option in sympyt funciton to not include lattice constraint
    #TODO #7 Output directly usable in sympy_pytential
    return fcn, constraints


    




def ideal_mixing_term(cs, mu0, T = 300, rho = 1):
    #Check if mu0 is set and assert it is of the correct length
    if not isinstance(mu0, Matrix):
        mu0 = Matrix(mu0)
    xs = cs/sum(cs)
    RT = 8.314*T
    return rho*cs.dot(mu0 + RT*xs.applyfunc(ln))

def hyperelastic_term(kappa, vi, cs, V, method = 'NeoHookean'):
    # kappa is provided, apply a neo-Hookean elasticity model
    V0 = cs.dot(vi)
    delta = V/V0
    if method == "NeoHookean":
        f_el = V0*kappa/2*(ln(delta))**2
    #TODO: #6 insert other hyperelastic models?
    return f_el
