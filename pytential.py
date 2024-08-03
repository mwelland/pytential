"""
Base class of the pYtential package.

"""
# Defined by sympy functions or surrogates
# Tools to make composite and reduce dimensionality

#Todo: Currently material must be built using single core, then saved. Issue is with randomization of variables in material creation (from free_symbols) and elmination of doubles during composite function creation.
# Facilitate finding norm of hessian (for preconditioning), eigenvalues, and nullspace. Operates on Hessian
# Material creation should be separate function. Not redone by all processes. Centrallized process in case of distributed needs?

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
    #TODO: Incompressability constraints belong to the potential as an extension of 'normal' elasticity. How to include?

    def __init__(self, fcn, vars, grad=None, hess=None, differential_structure = None, constraints = None):
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

        self.fcn  = fcn
        self.vars = vars
        self.grad = grad
        self.hess = hess

        if self.differential_structure is None:
            pass
            #TODO: Complete this! return self.differential_structure( *args, **kwargs)
        else:
            self.differential_structure = differential_structure

        self.constraints = constraints
        #self.additional_fields = kwargs

        

    def field(self, *args, **kwargs):
        return self.fcn(*args, **kwargs)
    
    def gradient(self, *args, **kwargs):
        return self.grad( *args, **kwargs)
    
    def hessian(self, *args, **kwargs):
        return self.hess(*args, **kwargs) 
    
    def differential_structure(self,  *args, **kwargs):
        return self.differential_structure(*args, **kwargs)
    
    #**Method to evaluate gradient with certain components based on vars**


