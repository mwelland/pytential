from .utils import find_matching_vars, get_sum_constraint_expressions
import dill as pickle
from sympy import lambdify

"""
Base class of the pYtential package.

"""
# Defined by sympy functions or surrogates
# Tools to make composite and reduce dimensionality

# Todo: Currently material must be built using single core, then saved. Issue is with randomization of variables in material creation (from free_symbols) and elmination of doubles during composite function creation.
# Facilitate finding norm of hessian (for preconditioning), eigenvalues, and nullspace. Operates on Hessian
# Material creation should be separate function. Not redone by all processes. Centrallized process in case of distributed needs?

def args_to_list(func):
    """
    Decorator to convert keyword arguments to a vector or pass through a vector
    """
    def wrapper(self, *args, **kwargs):
        if len(args) == 1: #and isinstance(args[0], (list, np.ndarray)):
            # With one argument, assume a vector TODO: Check not a dictionary?
            args = args[0]
        elif kwargs and not args:
            # Function was passed a set of keywords. Order into a vector accoding to self.vars
            args = [kwargs.get(v, 0) for v in self.vars]
            #args = [kwargs[v] for v in self.vars]
        else:
            raise ValueError("Only one positional argument is allowed.")
        #print(args)
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

        assert all(isinstance(v, str) for v in vars), "Variables must be strings"

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

        result = 'Pytential of type ' + str(type(self)) + '\n' + \
             'Variables: ' + str(self.vars) + '\n' + \
             'Potential: ' + str(self.fcn) + '\n' + \
             'Gradient: ' + str(self.grad) + '\n' + \
             'Hessian: ' + str(self.hess) + '\n'

        print('cont', self.constraints)
        if self.constraints:
            result += 'Constraints: ' + str(self.constraints) + '\n'
    
        return result

    def find_matching_vars(self, pattern):
        """
        Class method for finding variables in a list of variables that match a pattern
        """

        return find_matching_vars(self.vars, pattern)
    
    def write_to_file(self, file_name):
        if not file_name.endswith('.pkl'):
            file_name += '.pkl'
        with open(file_name, 'wb') as output:
             pickle.dump(self, output)

    def read_potential(name):
        print('hi', name)
        if not name.endswith('.pkl'):
            name += '.pkl'
        with open(name, 'rb') as input:
            pot = pickle.load(input)
        return pot

    # def load_potential(file_name):
    #     from os.path import getmtime, isfile, dirname, join, basename, splitext
    #     from importlib import import_module, util

    #     def build_potential_from_file_path(file_path):
    #         mod_name = splitext(basename(file_path))[0]
    #         spec = util.spec_from_file_location(mod_name, file_path)
    #         potential_file = util.module_from_spec(spec)
    #         spec.loader.exec_module(potential_file)
    #         print('Building potential')
    #         return potential_file.build_potential()

    #     def load_or_build_potential_from_file(file_name):
    #         file_name_py = file_name +'.py'
    #         file_name_saved = file_name +'.pkl'

    #         # Handling mpi distribution: Potentials are loaded by all ranks, but are built on one rank.
    #         # Not ideal since it implies copies of potentials everywhere. Better to centralize...?

    #         # Ensure the saved potential is up to date.
    #         if isfile(file_name_py):
    #             if not isfile(file_name_saved) or getmtime(file_name_py) > getmtime(file_name_saved):
    #                 potential = build_potential_from_file_path(file_name_py)
    #                 potential.write_to_file(file_name_saved)

    #         #---->Check to ensure potential only being built on one rank. Needed?
    #         # from mpi4py import MPI
    #         # comm = MPI.COMM_WORLD
    #         # rank = comm.Get_rank()
    #         # if rank ==0:
    #         #     if isfile(file_name_py):
    #         #         if not isfile(file_name_saved) or getmtime(file_name_py) > getmtime(file_name_saved):
    #         #             potential = build_potential_from_file_path(file_name_py)
    #         #             potential.write_to_file(file_name_saved)
    #         # comm.barrier()
        
    #         if isfile(file_name_saved):
    #             return read_potential(file_name_saved)


    # module_path = dirname(__file__) if '__file__' in globals() else '.'
    # module_path = join(module_path, 'common_systems')
    # paths = ['.', module_path]

    # for path in paths:
    #     file_path = join(path, file_name)
    #     potential = load_or_build_potential_from_file(file_path)
    #     if potential is not None:
    #         return potential

    # raise FileNotFoundError("Potential not found")


    def add_sum_constraints(self, pattern_var_pairs):
        """
        Adds sum constraints to the pytential

        Args:
            pattern_var_pairs (list): A list of tuples, where each tuple contains a pattern and the corresponding variable to collect.
        """

        new_constraint_expressions = get_sum_constraint_expressions(self.vars, pattern_var_pairs)
        
        #NEED to update variables in case constraints contain new ones.  
        # Make immutable since variables are tied to positions

        self.constraints_sym += new_constraint_expressions
        lambdify_expr = lambda expr: lambdify([self.vars], expr, modules="scipy")
        self.constraints += [lambdify_expr(c) for c in new_constraint_expressions]

