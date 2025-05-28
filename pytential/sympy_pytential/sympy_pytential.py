from sympy import pprint, Matrix, lambdify, Expr, hessian, symbols
from .function_from_properties import function_from_properties #, sum_prefixed_variables
#from ..quadratic_pytential.matrix_methods import eliminate_linear_constraints
from .operations import expand_and_replace_variable_log_variable
from .. import pytential

#TODO: #11 Have sympy_pytential able to take a list of sympy expressions and output a list of pytentials. Maybe this belongs in the pytential class directly?



class sympy_pytential(pytential):
    """
    Make a pytential from a sympy expression

    Child class of 'pytential', with automatically populated field
    and retention of the sympy expression.

    Attributes:
        vars: an ordered list of the variable names as strings
        fcn:  function that returns to the scalar pytential
        grad: function that returns to the vector of 1st derivatives
        hess: function that returns to the matrix of second derivatives

    Future: Arguments may be any complete sets of conjugate variables
    """
    def __init__(self, fcn_sym, vars= None, constraints_sym = []):
        assert isinstance(fcn_sym, Expr), "Function is not a sympy expression." 

        # fcn_sym = expand_and_replace_variable_log_variable(fcn_sym)
        # print(fcn_sym)
        # Automatically populate vars, grad, and hess
        if vars is None:
            vars_fcn =  [l.name for l in fcn_sym.free_symbols]
            vars_constraints = [l.name for c in constraints_sym for l in c.free_symbols]
            vars = sorted(list(set(vars_fcn + vars_constraints)))
        grad_sym = Matrix([fcn_sym]).jacobian(vars)
        grad_sym = grad_sym.tolist()[0]
        hess_sym = hessian(fcn_sym, vars)
        hess_sym = hess_sym.tolist()
        structure_sym = [fcn_sym, grad_sym, hess_sym]
        
        def replace_log_with_log1p(expr):
            from sympy.codegen.cfunctions import log1p
            return expr.replace(sp.log, lambda arg: log1p(arg - 1))
        #TODO: #9 Replace log with log1p in the sympy expression for numerical stability. Add option to prevent. 

        lambdify_expr = lambda expr: lambdify([vars], expr, "numpy")

        fcn, grad, hess = [lambdify_expr(f) for f in structure_sym]
        differential_structure = lambdify_expr(structure_sym)

        constraints = [lambdify_expr(c) for c in constraints_sym]
        super().__init__(fcn, vars, grad=grad, hess=hess, differential_structure=differential_structure, constraints = constraints)
        # self.fcns = fcns
        self.fcn_sym = fcn_sym
        self.grad_sym = grad_sym
        self.hess_sym = hess_sym
        self.constraints_sym = constraints_sym

    # @classmethod
    # def from_properties(cls, properties, state = None, suffix = None):
    #     """
    #     Create a sympy_pytential from a dictionary of properties
    #     """
    #     f, constraints = function_from_properties(properties)

    #     if state is not None:
    #         f = f.subs(state)
    #         constraints = [c.subs(state) for c in constraints]

    #     pyt = sympy_pytential(f, constraints_sym=constraints)
    #     if suffix is not None:\
    #         pyt = pyt.append_to_variables(suffix)
        
    #     return pyt

       
    def pprint(self):
        """
        Pretty print the pytential and its gradients as sympy expressions - looks funny in jupyter?
        """
        print('\nVariables')
        print(self.vars)
        print('\nPotential')
        pprint(self.fcn_sym)
        print('\nGradient')
        pprint(self.grad_sym)
        print('\nHessian')
        pprint(self.hess_sym)

        if self.constraints_sym:
            print('\nConstraints')
            [pprint(c) for c in self.constraints_sym] 
    
    def __str__(self):
        """
        Pretty print the pytential and its gradients as sympy expressions
        """
        result = 'x = ' + str(self.vars) + '\n\n' + \
             'f(x) = ' + str(self.fcn_sym) + '\n\n' + \
             'f\'(x)= ' + str(self.grad_sym) + '\n\n' + \
             'f\"(x)= ' + str(self.hess_sym)
        # TODO: #12 Make 0= for all constraints listed
        if self.constraints_sym:
            result += '\n\nConstraints:\n0 = ' + '\n'.join([str(c) for c in self.constraints_sym]) + '\n'

        return result

    def __add__(self, other):
        """
        Adds two sympy pytentials
        """
        return sympy_pytential(self.fcn_sym + other.fcn_sym, constraints_sym = self.constraints_sym + other.constraints_sym)
    
    def rename_variables(self, variable_substitutions):
        """
        Renames variables in a sympy pytential according to the dictionary var_swap

        Args:
            variable_substitutions: a list of variable substitutions pairs
        """
        
        return sympy_pytential(self.fcn_sym.subs(variable_substitutions),  constraints_sym = [c.subs(variable_substitutions) for c in self.constraints_sym])

    def append_to_variables(self, suffix, variables_to_append=None):
        """
        Renames variables in a sympy pytential by appending a suffix

        Args:
            variables_to_append: a list of variable to append the suffix to
        """

        def warn_if_not_true(condition, message):
            if not condition:
                warnings.warn(message, UserWarning)

        if variables_to_append is None:
            variables_to_append = self.vars
        else:
            checked_variables = all(v in pytential.vars for v in variables_to_append)
            warn_if_not_true(checked_variables, "Not all variables are in the pytential")

        appended_variables = symbols(' '.join([str(v)+suffix for v in variables_to_append]))
        variables_to_rename = dict(zip(variables_to_append, appended_variables))
        return self.rename_variables(variables_to_rename)
    
    def get_sum_constraint_expressions(self, pattern_var_pairs):
        # USEFUL or only in making summation constraints?
        """
        Sums all terms in the expression that match a pattern, subtracts the variable to collect

        Args:
            pattern_var_pairs (list): A list of tuples, where each tuple contains a pattern and the corresponding variable to collect.

        Returns:
            list: A list of summed expressions for each pattern and variable pair.
        """

        return get_sum_constraint_expressions(self.vars, pattern_var_pairs)
    
    def add_constraints_sym(self, constraint_expressions):
        """
        Adds symbolic constraints to the pytential

        Args:
            constraints (list): A list of symbolic constraints
        """
        return sympy_pytential(self.fcn_sym,  constraints_sym = self.constraints_sym + constraint_expressions)
    
    def get_constraint_jacobian(self):
        """
        Returns the jacobian of the constraints with respect to the variables as a list of lists
        """
        return Matrix(self.constraints_sym).jacobian(self.vars).tolist()

    def set_variables(self, substitutions):
        """
        Substitutes variables in the pytential with new variables

        Args:
            substitutions: a list of variable substitutions pairs
        """
        
        return sympy_pytential(self.fcn_sym.subs(substitutions), constraints_sym = [c.subs(substitutions) for c in self.constraints_sym])

    # def quadratic_expansion(self, expansion_point):
    #     """
    #     returns a sympy pytential that is a quadratic expansion about the expansion point
    #     """
        
    #     hess = self.hess(**expansion_point)
    #     grad = self.grad(**expansion_point)
    #     f0 = self.fcn(**expansion_point)

    #     #TODO: Check that the expansion point is an equilibrium point based on equality of grad components (matched by variables?)
    #     return sympy_pytential.quadratic(hess=hess, grad=grad, f0=f0, vars=self.vars, constraints_sym = self.constraints_sym)
