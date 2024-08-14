"""
Base class of the pYtential package.

"""
# Defined by sympy functions or surrogates
# Tools to make composite and reduce dimensionality

#Todo: Currently material must be built using single core, then saved. Issue is with randomization of variables in material creation (from free_symbols) and elmination of doubles during composite function creation.
# Facilitate finding norm of hessian (for preconditioning), eigenvalues, and nullspace. Operates on Hessian
# Material creation should be separate function. Not redone by all processes. Centrallized process in case of distributed needs?

def args_to_list(func):
    """
    Decorator to convert keyword arguments to a vector or pass through a vector
    """
    def wrapper(self, *args, **kwargs):
        if len(args) == 1:
            # Only one positional argument is allowed
            args = args[0]
        elif kwargs and not args:
            # Map keyword arguments to a list
            args = [kwargs[v] for v in self.vars]
        else:
            raise ValueError("Only one positional argument is allowed.")
        #print('args in wrapper', args)
        return func(self, args)
    return wrapper

class pytential:
    """
    Interface to a thermodynamic pytential

    A potential has methods for the potential, gradient, and Hessian

    Attributes:
        vars: an ordered list of the variable names as strings
        fcn:  function that returns to the scalar potential
        grad: function that returns to the vector of 1st derivatives
        hess: function that returns to the matrix of second derivatives

    Future: Arguments may be any complete sets of conjugate variables
    """
    def __init__(self, fcn, vars, grad=None, hess=None, differential_structure = None, constraints = []):
        # variables - dictonary of {variable type: sympy variable}
        # fcns      - a list of potentials with arguments: variables

        """
        Class for a 'pytential'

        Args:
            fcn(x): a point to a function of (x)
            grad: gradient of fcn wrt x
            hess: hessian of fcn wrt x
            vars: an ordered list / dictionary of variables
            **kwargs: any additional fiels to be kept
        """

        # Flag or test for homogeneity? Test would be useful, flag not necessary unless there is value.
        # Can I automatically determine a set of homogenous variables / coordinates?

        self.vars = vars
        self._fcn  = fcn
        self._grad = grad
        self._hess = hess
        self._differential_structure = differential_structure

        # if self.differential_structure is None:
        #     pass
        #     #TODO: Complete this! return self.differential_structure( *args, **kwargs)
        # else:
        #     self.differential_structure = differential_structure

        self.constraints = constraints
        #self.additional_fields = kwargs
    
    #TODO: #9 in dictionary of args, accept wildcards. E.g.: c0*
    def __call__(self, *args, **kwargs):
        # Shorthand to field call
        return self.fcn(*args, **kwargs)

    @args_to_list
    def fcn(self, args):
        #print('args in fcn call', args)
        return self._fcn(args)
    
    @args_to_list
    def grad(self, *args, **kwargs):
        return self._grad( *args, **kwargs)
    
    @args_to_list
    def hess(self, *args, **kwargs):
        return self._hess(*args, **kwargs) 
    
    @args_to_list
    def differential_structure(self,  *args, **kwargs):
        return self._differential_structure(*args, **kwargs)
    
    #**Method to evaluate gradient with certain components based on vars**

    def __str__(self):
        """
        Pretty print the details of the pytential
        """
        return 'Pytential of type ' + str(type(self)) + '\n' + \
            'Variables: ' + str(self.vars) + '\n' + \
            'Potential: ' + str(self.fcn) + '\n' + \
            'Gradient: ' + str(self.grad) + '\n' + \
            'Hessian: ' + str(self.hess) + '\n' + \
            'Constraints: ' + str(self.constraints) + '\n'
        # print('\nPytential of type ' + str(type(self)))
        # print('\nVariables')
        # print(self.vars)
        # print('\nPotential')
        # pprint(self.fcn_sym)
        # print('\nGradient')
        # pprint(self.grad_sym)
        # print('\nHessian')
        # pprint(self.hess_sym)
        # print('\nConstraints')
        # pprint(self.constraints_sym)    
