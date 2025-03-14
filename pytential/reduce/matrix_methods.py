import numpy as np
import scipy.linalg as la


# TODO: specify variable names to keep by name, not index number. 
def reduce_qp(Q, c, A, b , dependent_indices):
    """
    Reduces an equality-constrained quadratic program by eliminating dependent variables.
    1/2 x^T Q x + c^T x, subject to A x = b 
    is reduced to
    1/2 x^T Q_tilde x + c_tilde^T x, subject to A_tilde x = b_tilde,
    where Q_tilde and c_tilde are the reduced quadratic and linear terms, respectively. 

    Parameters:
        Q (numpy.ndarray): Quadratic term matrix (symmetric).
        c (numpy.ndarray): Linear term vector.
        A (numpy.ndarray): Equality constraint matrix.
        b (numpy.ndarray): Equality constraint vector.
        dependent_indices (list): Indices of dependent variables to be eliminated.

    Returns:
        tuple: Reduced quadratic matrix (Q_tilde), reduced linear term (c_tilde).
    """

    # Identify indices of free variables
    all_indices = np.arange(Q.shape[0])
    free_indices = np.setdiff1d(all_indices, dependent_indices)

    # Partition matrices and vectors
    Q_dd = Q[np.ix_(dependent_indices, dependent_indices)]
    Q_df = Q[np.ix_(dependent_indices, free_indices)]
    Q_fd = Q[np.ix_(free_indices, dependent_indices)]
    Q_ff = Q[np.ix_(free_indices, free_indices)]

    c_d = c[dependent_indices]
    c_f = c[free_indices]

    A_d = A[:, dependent_indices]
    A_f = A[:, free_indices]

    A_d_cond = np.linalg.cond(A_d)
    if  A_d_cond > 1e10 or A_d_cond < 1e-10:
      print("Warning: A_d has a poor condition number, results may be inaccurate.")
      print("Consider eliminating another variable.")



    # Compute the pseudo-inverse of A_d
    A_d_inv = la.pinv(A_d, rcond=1e-15)

    A_d_inv_A_f = A_d_inv @ A_f
    A_d_inv_b = A_d_inv @ b

    # A_d_inv_A_f = la.lstsq(A_d, A_f)[0]
    # A_d_inv_b = la.lstsq(A_d, b)[0]
    # Q_dd = np.round(Q_dd, decimals=8)

    # Compute the reduced quadratic and linear terms
    Q_tilde = Q_ff - Q_fd @ A_d_inv_A_f - A_d_inv_A_f.T @ Q_df + A_d_inv_A_f.T @ Q_dd @ A_d_inv_A_f
    c_tilde = c_f - A_d_inv_A_f.T @ c_d - A_d_inv_A_f.T @ Q_dd @ A_d_inv_b + 0.5* (Q_fd) @ A_d_inv_b + 0.5* Q_df.T @ A_d_inv_b


    return Q_tilde, c_tilde






def equality_qp_dual(B, A, c=None, d=None):
    """
    Solve the equality-constrained quadratic program using the dual (range-space) method:
    
         minimize   (1/2)x^T B x + c^T x
         subject to A x = d,
    
    where B is a positive semidefinite (n x n) matrix and A is an (m x n) matrix.
    
    Parameters:
      B : (n x n) numpy array, assumed symmetric and positive semidefinite.
      A : (m x n) numpy array representing equality constraints.
      c : (n,) numpy array, the linear term (default is zero).
      d : (m,) numpy array, the right-hand side (default is zero).
    
    Returns:
      x : The solution vector in R^n.
      lam : The Lagrange multiplier vector in R^m.
    """

    B = np.array(B, dtype=np.float64)
    A = np.array(A, dtype=np.float64)

    n = B.shape[0]
    m = A.shape[0]
    
    if c is None:
        c = np.zeros(n)
    if d is None:
        d = np.zeros(m)
    
    # Compute the pseudoinverse of B.
    B_pinv = np.linalg.pinv(B, hermitian=True)

    # Form the dual system matrix:
    # M = A B^{+} A^T
    M = A.dot(B_pinv).dot(A.T)
    print(M)
    
    # Form the right-hand side for the dual system:
    # rhs_dual = - d - A B^{+} c
    rhs_dual = -d - A.dot(B_pinv).dot(c)
    
    # Solve for lambda (the Lagrange multipliers)
    lam, _, _, _ = np.linalg.lstsq(M, rhs_dual)
    print(lam)
    # Recover the primal variable x:
    x = -B_pinv.dot(c + A.T.dot(lam))
    
    return x, lam



def equality_qp(B, A, c=None, d=None):
    """
    Solve the equality-constrained quadratic program:
    
         minimize   (1/2)x^T B x + c^T x
         subject to A x = d,
    
    where B is a positive semidefinite (n x n) matrix, A is an (m x n) matrix,
    c is an (n,) vector, and d is an (m,) vector.
    
    Parameters:
      B : (n x n) numpy array (symmetric, positive semidefinite)
      A : (m x n) numpy array representing equality constraints
      c : (n,) numpy array, linear term (default is zero)
      d : (m,) numpy array, right-hand side of constraints (default is zero)
      
    Returns:
      x      : The optimizer (n-dimensional vector)
      lambda_: The Lagrange multipliers (m-dimensional vector)
    """

    B = np.array(B, dtype=np.float64)
    A = np.array(A, dtype=np.float64)
    
    n = B.shape[0]
    m = A.shape[0]
    
    # Set default values if not provided.
    if c is None:
        c = np.zeros(n)
    if d is None:
        d = np.zeros(m)
    
    # Build the KKT matrix:
    #    [  B   A^T ]
    #    [  A    0  ]
    KKT = np.block([[B, A.T],
                    [A, np.zeros((m, m))]])
    
    Mi = np.linalg.pinv(KKT)
    
    if not np.allclose(Mi, Mi.T):
        print('Warning: KKT inverse is not symmetric')
        #TODO: Make M symmetric?
    # Splice up result into response and minimizer
    
    Mi00 = Mi[:-m,:-m]
    Mi10 = Mi[-m:,:-m]
    Mi01 = Mi10.T
    Mi11 = Mi[-m:,-m:]

    G = Mi11
    mu0=Mi10@c
    #print('G\n',G)
    #print('mu0\n',mu0)
    #print('b\n', b)
    #G = sp.nsimplify(G,tolerance=1e-15)
    #response = G@x + Mi10@b
    fcn_q = lambda x: np.einsum('ij,ij->i', -.5*(x@G) + mu0, x)# (-.5*x@G+mu0)@x.T #+ mu0
    response = lambda x: -G@x + mu0
    minimizer = lambda x: -Mi00@c + Mi10@x.T

    return fcn_q, response, minimizer, G


    # # Right-hand side:
    # rhs = -np.concatenate([c, d])
    
    # # Solve the KKT system in a least-squares sense.
    # sol, residuals, rank, s = np.linalg.lstsq(KKT, rhs, rcond=None)
    
    # # Extract x and the Lagrange multipliers.
    # x = sol[:n]
    # lambda_ = sol[n:]
    
    # return x, lambda_
