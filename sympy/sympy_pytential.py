from sympy import pprint, Matrix, lambdify, Expr, hessian
from .function_from_properties import function_from_properties
#from .operations import append_to_sympy_variables
from .. import pytential

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

        assert isinstance(fcn_sym, Expr), "Function is not a sympy expression. Use 'pytential'"

        # Automatically populate vars, grad, and hess
        if vars is None:
            vars =  [l.name for l in fcn_sym.free_symbols]
      
        grad_sym = Matrix([fcn_sym]).jacobian(vars)
        grad_sym.simplify()
        grad_sym = grad_sym.tolist()[0]
                
        hess_sym = hessian(fcn_sym, vars)
        hess_sym.simplify()

        structure_sym = [fcn_sym, grad_sym, hess_sym]
        
        lambdify_expr = lambda expr: lambdify(vars, expr, modules="scipy")

        fcn, grad, hess = [lambdify_expr(f) for f in structure_sym]
        differential_structure = lambdify_expr(structure_sym)

        constraints = [lambdify_expr(c) for c in constraints_sym]

        super().__init__(fcn, vars, grad=grad, hess=hess, differential_structure=differential_structure, constraints = constraints)

        #TODO: Would this be faster if I 

        # self.fcns = fcns
        self.fcn_sym = fcn_sym
        self.grad_sym = grad_sym
        self.hess_sym = hess_sym
        self.constraints_sym = constraints_sym


    def __str__(self):
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
        print('\nConstraints')
        pprint(self.constraints_sym)

        return ''

def sympy_pytential_from_properties(properties, suffix= None):
    f, constraints = function_from_properties(properties, T = 300)
    pyt = sympy_pytential(f, constraints_sym=[constraints])
    # if suffix is not None:
    #     pyt = append_to_sympy_variables(pytential, suffix)
    return pyt

