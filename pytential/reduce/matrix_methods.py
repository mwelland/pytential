import numpy as np

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
