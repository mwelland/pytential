from sympy import pprint, Matrix, lambdify, Expr, hessian
from .. import potential


class sympy_potential(potential):
    """
    Make a potential from a sympy expression

    Child class of 'potential', with automatically populated field
    and retention of the sympy expression.

    Attributes:
        vars: an ordered list of the variable names as strings
        fcn:  function that returns to the scalar potential
        grad: function that returns to the vector of 1st derivatives
        hess: function that returns to the matrix of second derivatives

    Future: Arguments may be any complete sets of conjugate variables
    """
    #TODO: Incompressability constraints belong to the potential as an extension of 'normal' elasticity. How to include?

    def __init__(self, fcn_sym, vars= None):

        assert isinstance(fcn_sym, Expr), "Function is not a sympy expression. Use 'potential'"

        # Automatically populate vars, grad, and hess
        if vars is None:
            vars =  [l.name for l in fcn_sym.free_symbols]
        print(vars)

        # if phase_id:
        #     vars_old = vars
        #     vars = [v+'_'+ phase_id for v in vars_old]
        #     fcn_sym = fcn_sym.subs(zip(vars_old, vars))
        print(Matrix([fcn_sym]).jacobian(vars))

        # fcns_sym = [fcn_sym, 
        #             Matrix([fcn_sym]).jacobian(vars).tolist()[0], 
        #             hessian(fcn_sym, vars).simplify(),
        #             ] 
        
        grad_sym = Matrix([fcn_sym]).jacobian(vars)
        grad_sym.simplify()
        
        hess_sym = hessian(fcn_sym, vars)
        hess_sym.simplify()

        #fcn, grad, hess = [lambdify([vars], f, modules="scipy") for f in [fcn_sym, grad_sym, hess_sym]]

        fcn = lambdify(vars, fcn_sym, modules="scipy")     
        grad = lambdify(vars, grad_sym.tolist()[0], modules="scipy")
        hess = lambdify(vars, hess_sym, modules="scipy")

        # Create a 'potential' with the calculated functions
        super().__init__(fcn, vars, grad=grad, hess=hess)

        # self.fcns = fcns
        self.fcn_sym = fcn_sym
        self.grad_sym = grad_sym
        self.hess_sym = hess_sym


    def __str__(self):
        """
        Pretty print the potential and its gradients as sympy expressions
        """
        print('\nVariables')
        print(self.vars)
        print('\nPotential')
        pprint(self.fcn_sym)
        print('\nGradient')
        pprint(self.grad_sym)
        print('\nHessian')
        pprint(self.hess_sym)
        return ''

    def rename_vars(self, suffix):
        # Add 'suffix' to all variables
        #vars_part = None
        vars_old = self.vars
        self.vars = [v+'_'+suffix for v in vars_old]
        replacement_dict = {v_old: v_new for v_old, v_new in zip(vars_old, self.vars)}
        self.fcn_sym = self.fcn_sym.subs(replacement_dict)
        self.grad_sym = self.grad_sym.subs(replacement_dict)
        self.hess_sym = self.hess_sym.subs(replacement_dict)