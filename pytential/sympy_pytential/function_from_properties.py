from sympy import Matrix, symbols, ln

def function_from_properties(properties):
    #Returns a sympy function consistent with the properties dictionary
    
    assert type(properties) is dict, "properties must be a dictionary"

    T, V = symbols('T V')

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
    constraints = []
    
    
    cs = Matrix(symbols('c:{}'.format(n)))

    fcn += ideal_mixing_term(cs, properties['mu0'], T = T, rho = properties['rho'])

    # If 'kappa' is defined, assume hyperelastic. Else, lattice constraints.
    if 'kappa' in keys:
        fcn += hyperelastic_term(
            kappa = properties['kappa'], 
            vi = properties['vi'], 
            cs = cs, 
            V = V)
    else:
        constraints += [cs.dot(properties['vi'])-V]
    #TODO: #5 option in sympyt funciton to not include lattice constraint
    #TODO #7 Output directly usable in sympy_pytential
    return fcn, constraints


def sum_prefixed_variables(vars, prefix): 
    #Forms the constraint for an extensive variable by matching any varialbes
    # in pyt. Useful to sum variables with a suffix.
    print('vars', vars)
    print('prefix', prefix)   
    def spfx(vars, prefix): 
        partial_variables = [s for s in vars if s.startswith(prefix)]    
        return sum(symbols(partial_variables))-symbols(prefix)
    return [spfx(vars, p) for p in prefix]


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
