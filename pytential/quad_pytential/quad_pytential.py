from sympy import pprint, Matrix, lambdify, Expr, hessian, symbols
from .matrix_methods import eliminate_linear_constraints, reduce_unconstrained_quadratic_by_minimization
from ..sympy_pytential import sympy_pytential

class quad_pytential(sympy_pytential):
    """
    A specialization of the sympy_pytential class for quadratic pytentials.
    This class is designed to handle quadratic pytentials, which are defined by a constant, symmetric Hessian matrix.

    Attributes:
        vars: an ordered list of the variable names as strings
        fcn:  function that returns to the scalar pytential
        grad: function that returns to the vector of 1st derivatives
        hess: function that returns to the matrix of second derivatives

    Future: Arguments may be any complete sets of conjugate variables
    """
    def __init__(self, f0, b, Q,  vars, constraints_sym = []):
        """
        Initializes a quadratic pytential.
        Args:
            f0: constant term in the quadratic pytential
            b: linear term coefficients (vector)
            Q: Hessian matrix (symmetric)
            vars: list of variable names as strings
            constraints_sym: list of sympy expressions representing constraints
        """
        

        x = Matrix(symbols(vars))
        bs = Matrix(b)
        Qs = Matrix(Q)
        dx = x#-Matrix(x0)
        assert Qs.is_symmetric, "Hessian matrix Q must be symmetric."
        
        fcn = 1/2 * (dx.T * Qs * dx)[0, 0] + bs.dot(dx) + f0
        
        super().__init__(fcn, vars = vars, constraints_sym = constraints_sym)

        self.f0 = f0
        self.b = b
        self.Q = Q
        self._hess = Q


        

    @classmethod
    def from_homog_pyt(cls, pyt, y0):
        """
        Creates a quadratic pytential from a pytential evaluated at y0.
        """
        x0 = pyt.dict_to_array(y0)
        return quad_pytential(f0 = 0,
                                b = pyt._grad(x0), 
                                Q = pyt._hess(x0), 
                                vars=pyt.vars,
                                constraints_sym=pyt.constraints_sym)
    

    def hess(self, *args, **kwargs):
        return self._hess # Return the stored Hessian matrix

    def reduce_by_eliminating_linear_constraints(self, vars_to_keep):
        """
        Removes linear constraints through nullspace projection.
        Args:
            vars_to_keep: list of variable names to keep in the reduced pytential       
        """
        print(self.vars)
        print(vars_to_keep)

        free_indices = [self.vars.index(var) for var in vars_to_keep]# self.vars[i] for i in vars_to_keep]
        print(free_indices)

        A, c = self.get_constraint_matrix_and_vector()

        Q, b, f0, new_independent_vars_original_indices, T_map_for_reconstruction, t_offset_for_reconstruction = eliminate_linear_constraints(self.Q, self.b, self.f0, A, c, free_indices)
        
        
        vars_out = [self.vars[i] for i in new_independent_vars_original_indices.astype(int)]
        
        return quad_pytential(f0=f0, b = b, Q = Q, vars = vars_out)
        #return sympy_pytential.quadratic(hess=lambda_linear, grad=lambda_const, f0=0, vars = vars_to_keep)
    
    def reduce_uncontrained_by_minimization(self, vars_to_keep):
        
        free_indices = [self.vars.index(var) for var in vars_to_keep]# self.vars[i] for i in vars_to_keep]
        print(free_indices)
        Q, b, f0, kept_indices_uc_relative, T_map_for_reconstruction, t_offset_for_reconstruction = reduce_unconstrained_quadratic_by_minimization(self.Q, self.b, self.f0, free_indices)

        vars_out = [self.vars[i] for i in kept_indices_uc_relative]

        return quad_pytential(f0=f0, b = b, Q = Q, vars = vars_out)
    
    def print_eigensystem(self):
        """
        Calculates the eigenvalues and eigenvectors of the Hessian matrix.
        Returns:
            tuple: (eigenvalues, eigenvectors)
        """
        import numpy as np
        hess = np.asarray(self._hess, dtype=float)
        eigenvalues, eigenvectors = np.linalg.eig(hess)

        print("Eigenvalues:\n", eigenvalues)
        print("Eigenvectors:\n", eigenvectors)

        # # Print eigenvalues and corresponding eigenvectors
        # for i in range(len(eigenvalues)):
        #     print(f"Eigenvalue: {eigenvalues[i]}")
        #     print(f"Eigenvector: {eigenvectors[i][0]}")  # Access the eigenvector
        #     print("-" * 20)


