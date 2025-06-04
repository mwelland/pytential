from .utils import find_matching_vars, get_sum_constraint_expressions
import dill as pickle
import numpy as np
#from sympy import lambdify

"""
Base class of the pYtential package.

"""

#TODO: Currently calling is execting a vector of arguments. If there in only one variable, can this be skipped?

# Todo: Currently material must be built using single core, then saved. Issue is with randomization of variables in material creation (from free_symbols) and elmination of doubles during composite function creation.
# Facilitate finding norm of hessian (for preconditioning), eigenvalues, and nullspace. Operates on Hessian
# Material creation should be separate function. Not redone by all processes. Centrallized process in case of distributed needs?


def args_to_list(func):
    """
    Decorator to allow flexibility in specifying arguments.

    Ensures functions receive an n x p array, where n = number of variables in self.vars and p = number of points to evaluate. 

    - a single argument with elements in the order in self.vars
    - If keyword arguments are given, they are converted to an array in the order of self.vars.
    - Scalars in kwargs are broadcast to match the length of the longest array.
    - All arrays/lists must be of length 1 or the same length.
    - Only one positional argument is allowed.
    """
    def wrapper(self, *args, **kwargs):
        if len(args) == 1: #and not kwargs:
            arg_vec = np.asarray(args[0])
            arg_vec = arg_vec.reshape(len(self.vars),-1)    #Ensures the first dimension is the number of variables
        elif not args and kwargs:
            arg_vec = self.dict_to_array(kwargs)
        else:
            raise ValueError("Only one positional argument (vector) or keyword arguments allowed.")
        return func(self, arg_vec)
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
        # All of these functions have vector arguments
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

    def dict_to_array(self, dict_of_vars, check_extra_keys=False):
        """
        # Takes a dictionary of variables and returns an array according to the order of self.vars
        """
        if check_extra_keys:
            extra_keys = set(dict_of_vars.keys()) - set(self.vars)
            if extra_keys:
                raise ValueError(f"Unknown keyword arguments: {extra_keys}. Allowed: {self.vars}")
        
        # Check if all variables have been assigned a value
        missing_vars = set(self.vars) - set(dict_of_vars.keys())
        if missing_vars:
            raise ValueError(f"Missing values for variables: {missing_vars}.  Must provide values for all of: {self.vars}")
        
        values = [dict_of_vars.get(v, 0) for v in self.vars]        
        lengths = [np.size(val) for val in values]
        if all(l == 1 for l in lengths):
            # All values are scalars, so we can just return a 1D array
            return np.array(values)
        else:
            # # Check if all values are scalars or arrays of the same length
            # if not all(l == 1 or l == lengths[0] for l in lengths):
            #     raise ValueError(
            #         f"All arguments must be scalars or arrays of the same length. Got lengths: {lengths}"
            #     )
            # Determine the target length for broadcasting
            target_len = max(lengths)
            broadcasted = [np.full(target_len, val) if np.size(val) == 1 else np.asarray(val) for val in values]
            return np.vstack(broadcasted)
    
    def vars_to_indices(self, vars_to_find):
        """
        Returns the indices of the variables in self.vars that match vars_to_find

        Args:
            vars_to_find (list): A list of variable names to find in self.vars

        Returns:
            list: A list of indices corresponding to vars_to_find in self.vars
        """
        return [self.vars.index(v) for v in vars_to_find if v in self.vars]
    
    def find_matching_vars(self, pattern):
        """
        Class method for finding variables in a list of variables that match a pattern
        """

        return find_matching_vars(self.vars, pattern)
    
    def set_variables(self, substitutions):
        """
        Set the variables in the potential to a new set of variables

        Args:
            substitutions (dict): A dictionary of substitutions to be made in the potential
        """
        print("Not implemented. Overridden by sympy_pytential")
        pass
    
    def get_constraint_matrix_and_vector(self):
        """
        Computes the constraint matrix (Jacobian) and constants
        Ax=b

        Returns:
            tuple: (matrix, constants)
                - matrix: A 2D NumPy array where each row is the vector from a constraint.
                - constants: A 1D NumPy array of constants from each constraint.
        """
        n = len(self.vars)
        matrix = []
        constants = []

        for fcn in self.constraints:
            # Compute the constant and vector for the current constraint
            constant = fcn(np.zeros((n, 1)))
            line = fcn(np.identity(n)) - constant

            # Append to the matrix and constants list
            matrix.append(line)
            constants.append(constant)

        # Convert to NumPy arrays
        matrix = np.array(matrix)
        constants = np.array(constants)

        return matrix, -constants

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


    # def add_sum_constraints(self, pattern_var_pairs):
    #     """
    #     Adds sum constraints to the pytential

    #     Args:
    #         pattern_var_pairs (list): A list of tuples, where each tuple contains a pattern and the corresponding variable to collect.
    #     """

    #     new_constraint_expressions = get_sum_constraint_expressions(self.vars, pattern_var_pairs)
        
    #     #NEED to update variables in case constraints contain new ones.  
    #     # Make immutable since variables are tied to positions

    #     self.constraints_sym += new_constraint_expressions
    #     lambdify_expr = lambda expr: lambdify([self.vars], expr, modules="scipy")
    #     self.constraints += [lambdify_expr(c) for c in new_constraint_expressions]

