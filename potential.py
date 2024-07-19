"""
Base class of the pYtential package.

"""
# Defined by sympy functions or surrogates
# Tools to make composite and reduce dimensionality

#Todo: Currently material must be built using single core, then saved. Issue is with randomization of variables in material creation (from free_symbols) and elmination of doubles during composite function creation.
# Facilitate finding norm of hessian (for preconditioning), eigenvalues, and nullspace. Operates on Hessian
# Material creation should be separate function. Not redone by all processes. Centrallized process in case of distributed needs?

class potential:
    """
    Interface to a thermodynamic potential

    A potential has methods for the potential, gradient, and Hessian

    Attributes:
        vars: an ordered list of the variable names as strings
        fcn:  function that returns to the scalar potential
        grad: function that returns to the vector of 1st derivatives
        hess: function that returns to the matrix of second derivatives

    Future: Arguments may be any complete sets of conjugate variables
    """
    #TODO: Incompressability constraints belong to the potential as an extension of 'normal' elasticity. How to include?

    def __init__(self, fcns, vars, constraints = None **kwargs):
        # variables - dictonary of {variable type: sympy variable}
        # fcns      - a list of potentials with arguments: variables

        """
        Class for a 'potential'

        Interface to a 'potential'. Provides an interface to evaluation.

        Args:
            fcn(x): a point to a function of (x)
            grad: gradient of fcn wrt x
            hess: hessian of fcn wrt x
            phase_id: An optional index of the phase
            **kwargs: any additional fiels to be kept
        """

        # Flag or test for homogeneity? Test would be useful, flag not necessary unless there is value.
        # Can I automatically determine a set of homogenous variables / coordinates?

        # Need to move away from sympy reliance. Causing problems with: vectorized evaluation / plotting
        # Keep for forming, then remove.

        self.fcn  = fcn
        self.vars = vars
        self.grad = grad
        self.hess = hess
        self.constraints = constraints
        self.additional_fields = kwargs

    # def rename_vars(self, suffix):
    #     # Add 'suffix' to all variables
    #     #vars_part = None
    #     self.vars = [v+'_'+suffix for v in self.vars]

# def write_to_file(file, file_name):
#     import dill as pickle
#     if not file_name.endswith('.pkl'):
#         file_name += '.pkl'
#     with open(file_name, 'wb') as output:
#          pickle.dump(self, output)