import numpy as np
import scipy.linalg as la
from scipy.linalg import qr, solve_triangular

DEFAULT_TOL = 1e-8

def _check_symmetry(Q, name="Q_matrix", tol=DEFAULT_TOL):
    """Checks if a matrix Q is square and symmetric."""
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix. Got shape {Q.shape}")
    if not np.allclose(Q, Q.T, atol=tol, rtol=tol):
        raise ValueError(f"{name} must be a symmetric matrix.")

def _validate_and_derive_indices(num_total_vars, priority_kept_indices):
    """Validates priority_kept_indices and derives elim_candidate_indices."""
    priority_kept_indices = np.array(sorted(list(set(priority_kept_indices))), dtype=int) # Sort and unique

    if num_total_vars < 0:
        raise ValueError("num_total_vars cannot be negative.")
    if priority_kept_indices.size > 0:
        if not (priority_kept_indices.max() < num_total_vars and priority_kept_indices.min() >= 0):
            raise ValueError("priority_kept_indices are out of bounds.")
    elif priority_kept_indices.size > num_total_vars : # Catches empty but num_total_vars = 0 too
        raise ValueError("priority_kept_indices cannot have more elements than num_total_vars.")


    all_original_indices = np.arange(num_total_vars)
    elim_candidate_indices = np.setdiff1d(all_original_indices, priority_kept_indices, assume_unique=True)
    
    n_kept = len(priority_kept_indices)
    n_elim_cand = len(elim_candidate_indices)

    return priority_kept_indices, elim_candidate_indices, n_kept, n_elim_cand

def _preprocess_constraints_system(A_full, b_full, tol=DEFAULT_TOL):
    # Ensuring A_prime is full row rank and consistent)
    
    if A_full is None or A_full.size == 0:
        return np.empty((0, 0)), np.empty(0)
    
    A_full = np.asarray(A_full)
    if A_full.ndim == 1 and A_full.shape[0] > 0: A_full = A_full.reshape(1, -1)

    b_flat = np.asarray(b_full).flatten() if b_full is not None else np.zeros(A_full.shape[0])

    if A_full.shape[0] != b_flat.shape[0]:
        raise ValueError(f"A_full rows ({A_full.shape[0]}) != b_full elements ({b_flat.shape[0]}).")

    num_total_vars = A_full.shape[1]
    if num_total_vars == 0:
        if A_full.shape[0] > 0 and np.any(np.abs(b_flat) > tol):
            raise ValueError("Constraints Ax=b inconsistent (no variables, b non-zero).")
        return np.empty((0,0)), np.empty(0)
            
    b_col = b_flat.reshape(-1, 1)

    print('A', A_full)

    rank_A = np.linalg.matrix_rank(A_full, tol=tol)
    Ab_stack = np.hstack([A_full, b_col]) if A_full.shape[1] > 0 else b_col
    rank_Ab = np.linalg.matrix_rank(Ab_stack, tol=tol)
    if rank_A < rank_Ab:
        raise ValueError("Constraints A_full x = b_full inconsistent (rank(A) < rank([A|b])).")
    if rank_A == 0: 
        if np.any(np.abs(b_flat) > tol):
            raise ValueError("Constraints inconsistent (A zero-rank, b non-zero).")
        return np.empty((0, A_full.shape[1])), np.empty(0)
    
    # If A_full is full row rank, pass through
    if rank_A == A_full.shape[0]:  
        return A_full, b_flat

    _Q_at, _R_at, P_at_cols = qr(A_full.T, pivoting=True, mode='economic')
    independent_row_indices = sorted(P_at_cols[:rank_A])
    A_prime = A_full[independent_row_indices, :]
    b_prime = b_flat[independent_row_indices]
    return A_prime, b_prime

def _build_transformation_for_algebraic_elim(
    num_total_vars, kept_indices, elim_dep_indices, elim_ind_indices,
    S_dep_k, S_dep_ind, t_dep_const):
    """
    Builds T_map, t_offset such that x_original = T_map @ x_new + t_offset
    where x_new = [x_k^T, x_elim_ind^T]^T
    """
    n_k = len(kept_indices)
    n_elim_dep = len(elim_dep_indices)
    n_elim_ind = len(elim_ind_indices)
    n_new_vars = n_k + n_elim_ind

    T_map = np.zeros((num_total_vars, n_new_vars))
    t_offset = np.zeros(num_total_vars)

    # Mapping for x_k
    if n_k > 0:
        T_map[np.ix_(kept_indices, np.arange(n_k))] = np.eye(n_k)

    # Mapping for x_elim_ind
    if n_elim_ind > 0:
        T_map[np.ix_(elim_ind_indices, np.arange(n_k, n_new_vars))] = np.eye(n_elim_ind)

    # Mapping for x_elim_dep
    if n_elim_dep > 0:
        # x_elim_dep = S_dep_k @ x_k + S_dep_ind @ x_elim_ind + t_dep_const
        if n_k > 0:
            T_map[np.ix_(elim_dep_indices, np.arange(n_k))] = S_dep_k
        if n_elim_ind > 0:
            T_map[np.ix_(elim_dep_indices, np.arange(n_k, n_new_vars))] = S_dep_ind
        t_offset[elim_dep_indices] = t_dep_const
        
    return T_map, t_offset

def compute_affine_constraint_elimination_map(
    A_full, b_full, priority_kept_indices, tol=DEFAULT_TOL):
    """
    Computes the affine map x_original = T_map @ x_new_independent + t_offset,
    where x_new_independent are variables remaining after algebraically eliminating
    others using constraints Ax=b. x_new_independent will contain all
    priority_kept_indices and any elim_candidate_indices that couldn't be solved for.

    Args:
        A_full (np.ndarray/None): Full constraint matrix (M x N_total).
        b_full (np.ndarray/None): Full constraint RHS vector (M,).
        priority_kept_indices (list/np.ndarray): Original indices of variables
                                                 that must be in x_new_independent.
        tol (float): Tolerance for rank calculations.

    Returns:
        tuple: (T_map, t_offset, new_independent_vars_original_indices)
            T_map (np.ndarray): Transformation matrix (N_total x N_new_independent).
            t_offset (np.ndarray): Offset vector (N_total,).
            new_independent_vars_original_indices (np.ndarray): 1D array of original
                indices corresponding to the variables in x_new_independent.
    """

    A_full = np.asarray(A_full, dtype=np.float64) if A_full is not None else None
    b_full = np.asarray(b_full).flatten() if b_full is not None else np.zeros(A_full.shape[0]) if A_full is not None else np.array([])

    num_total_vars = A_full.shape[1] if A_full is not None and A_full.size > 0 else 0
    if num_total_vars == 0 and not priority_kept_indices : # No variables, no kept indices
        return np.empty((0,0)), np.empty(0), np.array([], dtype=int)
    if num_total_vars == 0 and priority_kept_indices:
        raise ValueError("priority_kept_indices specified but A_full implies no variables.")


    priority_kept_indices, elim_candidate_indices, n_kept, n_elim_cand = \
        _validate_and_derive_indices(num_total_vars, priority_kept_indices)

    A_prime, b_prime_flat = _preprocess_constraints_system(A_full, b_full, tol)

    if A_prime.size == 0: # No effective constraints
        # All original variables remain independent
        new_independent_vars_original_indices = np.arange(num_total_vars)
        T_map = np.eye(num_total_vars)
        t_offset = np.zeros(num_total_vars)
        return T_map, t_offset, new_independent_vars_original_indices

    # Partition A_prime based on priority_kept and elim_candidate indices
    A_kept_prime = A_prime[:, priority_kept_indices] if n_kept > 0 else np.empty((A_prime.shape[0], 0))
    A_elim_cand_prime = A_prime[:, elim_candidate_indices] if n_elim_cand > 0 else np.empty((A_prime.shape[0], 0))

    if n_elim_cand == 0: # Nothing to eliminate candidates for
        # All variables are effectively "kept". new_independent are just priority_kept_indices.
        # This means A_elim_cand_prime is empty.
        # Remaining constraints: A_kept_prime @ x_kept = b_prime_flat
        # This function is about variable elimination, if no elim_candidates, T_map is identity for kept,
        # but it implies there are constraints on kept_indices which this function does not "solve" for.
        # The output x_new_independent ARE the kept_indices.
        # If A_kept_prime is not empty, it means there are constraints only on kept vars.
        # This function's role is to define x_orig in terms of a new set of *fewer or equal* independent vars.
        # If all are kept, or no elim_cand, then T_map=Identity, t_offset=0, new_indep=original.
        # The constraints A_kept_prime @ x_kept = b_prime_flat are *not resolved* by this function,
        # they would persist if this was part of a larger problem.
        # For this specific function, if n_elim_cand = 0, all vars are independent in this step.
        if np.linalg.matrix_rank(A_kept_prime, tol=tol) > 0:
             print("Warning: Constraints exist only among priority_kept_indices. "
                   "These constraints are not eliminated by this function. "
                   "The transformation returned will be an identity map for these variables.")

        new_independent_vars_original_indices = np.arange(num_total_vars) # All original vars are "independent" from elim perspective
        T_map = np.eye(num_total_vars)
        t_offset = np.zeros(num_total_vars)
        return T_map, t_offset, new_independent_vars_original_indices


    # Perform QR with column pivoting on A_elim_cand_prime to find dependent/independent elim_cand variables
    # A_elim_cand_prime @ P_ec = Q_ec @ R_ec
    Q_ec, R_ec, P_ec_cols = qr(A_elim_cand_prime, pivoting=True, mode='economic')
    rank_A_ec = np.linalg.matrix_rank(A_elim_cand_prime, tol=tol) # Number of dependent elim_cand vars

    # Permuted original indices of elimination candidates
    permuted_elim_cand_indices = elim_candidate_indices[P_ec_cols]
    
    elim_dep_orig_indices = permuted_elim_cand_indices[:rank_A_ec]
    elim_ind_orig_indices = permuted_elim_cand_indices[rank_A_ec:]
    n_elim_dep = len(elim_dep_orig_indices)
    n_elim_ind = len(elim_ind_orig_indices)

    # R_ec = [R11 R12]
    #        [ 0  R22] (R22 is zero block if rank_A_ec < num_constraints)
    R11 = R_ec[:rank_A_ec, :rank_A_ec]
    R12 = R_ec[:rank_A_ec, rank_A_ec:] # Will be empty if rank_A_ec = n_elim_cand

    # Effective RHS: b_eff = Q_ec.T @ (b_prime_flat - A_kept_prime @ x_kept)
    # Let b_eff_const_part = Q_ec.T @ b_prime_flat
    # Let b_eff_xk_part_coeff = - Q_ec.T @ A_kept_prime
    
    b_eff_const_part_col = Q_ec.T @ b_prime_flat.reshape(-1,1)
    b_eff_xk_part_coeff = -Q_ec.T @ A_kept_prime if n_kept > 0 else np.zeros((Q_ec.shape[1], n_kept))


    # Check for inconsistencies revealed by QR (bottom rows of Q_ec.T @ (b_prime - ...))
    # These are constraints like 0 = const or 0 = f(x_k).
    # If rank_A_ec < Q_ec.shape[0] (num_constraints_prime):
    if rank_A_ec < A_prime.shape[0]: # num_constraints_prime
        # Check if Q_ec.T @ (b_prime_flat - A_k_prime @ x_k) is zero for rows > rank_A_ec
        # This means bottom_const_part + bottom_xk_part @ x_k = 0 for all x_k.
        # Implies bottom_const_part = 0 AND bottom_xk_part = 0.
        # If not, it's an issue for _preprocess_constraints, or constraints implicitly on x_k.
        # For now, we assume _preprocess_constraints handled global consistency.
        # If constraints like 0 = f(x_k) arise, they are effectively new constraints
        # on the "independent" x_k, not resolved here. We proceed assuming 0=0.
        pass


    # Solve R11 @ x_elim_dep_p + R12 @ x_elim_ind_p = b_eff_const_part[:rank_A_ec] + b_eff_xk_part_coeff[:rank_A_ec] @ x_k
    # x_elim_dep_p = R11_inv @ (b_eff_const_part_dep - b_eff_xk_part_coeff_dep @ x_k - R12 @ x_elim_ind_p)
    
    S_dep_k = np.zeros((n_elim_dep, n_kept))
    if n_kept > 0 and rank_A_ec > 0:
        S_dep_k = solve_triangular(R11, b_eff_xk_part_coeff[:rank_A_ec, :], lower=False)
        
    S_dep_ind = np.zeros((n_elim_dep, n_elim_ind))
    if n_elim_ind > 0 and rank_A_ec > 0: # If there are independent elim_cand variables
        S_dep_ind = -solve_triangular(R11, R12, lower=False)
        
    t_dep_const = np.zeros(n_elim_dep)
    if rank_A_ec > 0:
         t_dep_const = solve_triangular(R11, b_eff_const_part_col[:rank_A_ec,:], lower=False).flatten()


    # New independent variables are priority_kept_indices and elim_ind_orig_indices
    new_independent_vars_original_indices = np.concatenate((priority_kept_indices, elim_ind_orig_indices)).astype(int)
    
    T_map, t_offset = _build_transformation_for_algebraic_elim(
        num_total_vars, priority_kept_indices, elim_dep_orig_indices, elim_ind_orig_indices,
        S_dep_k, S_dep_ind, t_dep_const
    )
        
    return T_map, t_offset, new_independent_vars_original_indices

def eliminate_linear_constraints(
    Q_full, c_full, k0_full, A_full, b_full, 
    priority_kept_indices, tol=DEFAULT_TOL):
    """
    Transforms a constrained quadratic problem into an unconstrained one
    by algebraically eliminating variables using linear constraints.
    The new unconstrained problem's variables will be priority_kept_indices and
    any other original variables that couldn't be eliminated.
    NO minimization is performed in this step. Q_full must be positive semidefinite.

    Args:
        Q_full (np.ndarray): PSD quadratic coefficient matrix (N_total x N_total).
        c_full (np.ndarray): Linear coefficient vector (N_total,).
        k0_full (float): Constant term.
        A_full (np.ndarray/None): Constraint matrix (M x N_total).
        b_full (np.ndarray/None): Constraint RHS vector (M,).
        priority_kept_indices (list/np.ndarray): Original indices of variables to ensure
                                                 are part of the new independent set.
        tol (float): Tolerance for rank determination and comparisons.

    Returns:
        tuple: (Q_new, c_new_flat, k0_new, new_independent_vars_original_indices, 
                T_map_for_reconstruction, t_offset_for_reconstruction)
            Q_new (np.ndarray): Quadratic matrix for the new unconstrained problem.
                                Variables correspond to new_independent_vars_original_indices.
            c_new_flat (np.ndarray): Linear vector for the new problem (1D).
            k0_new (float): Constant term for the new problem.
            new_independent_vars_original_indices (np.ndarray): 1D array of original indices
                that form the variables of the new unconstrained quadratic problem.
            T_map_for_reconstruction (np.ndarray): Matrix to reconstruct original variables:
                                                    x_orig = T_map @ x_new + t_offset.
            t_offset_for_reconstruction (np.ndarray): Offset for reconstruction.
    """
    Q_full = np.asarray(Q_full)
    c_full_flat = np.asarray(c_full).flatten()

    num_total_vars = Q_full.shape[0]

    if c_full_flat.shape[0] != num_total_vars:
        raise ValueError(f"c_full length must match Q_full dim.")
    
    _check_symmetry(Q_full, name="Q_full", tol=tol)
    
    T_map, t_offset, new_independent_vars_original_indices = \
        compute_affine_constraint_elimination_map(A_full, b_full, priority_kept_indices, tol)
    
    # Transform the quadratic objective:
    # f_new(x_new) = 0.5 * (T_map@x_new + t_offset)^T @ Q_full @ (T_map@x_new + t_offset) + 
    #                c_full^T @ (T_map@x_new + t_offset) + k0_full
    
    Q_new = T_map.T @ Q_full @ T_map
    
    c_new_col = T_map.T @ Q_full @ t_offset.reshape(-1,1) + T_map.T @ c_full_flat.reshape(-1,1)
    
    k0_new = (0.5 * t_offset.T @ Q_full @ t_offset + 
              c_full_flat.T @ t_offset + 
              k0_full)
    # Ensure k0_new is a scalar float if it ended up as a 1x1 array
    if isinstance(k0_new, np.ndarray): k0_new = k0_new.item()
              
    _check_symmetry(Q_new, name="Q_new (from algebraic elim)", tol=tol) # Should be symmetric

    return (Q_new, c_new_col.flatten(), float(k0_new), 
            new_independent_vars_original_indices, T_map, t_offset)

# def reduce_unconstrained_quadratic_by_minimization(
#     Q_uc_orig, c_uc_orig_flat, k0_uc_orig, 
#     kept_indices_uc_relative, elim_indices_uc_relative, # Indices relative to THIS problem's variables
#     tol=DEFAULT_TOL, rcond_pinv=RCOND_PINV):
#     """
#     Reduces dimensionality of an UNCONSTRAINED quadratic problem by minimizing out
#     a subset of its variables (elim_indices_uc_relative).
#     Q_uc_orig must be positive semidefinite.

#     Args:
#         Q_uc_orig (np.ndarray): PSD quadratic matrix of the input unconstrained problem.
#         c_uc_orig_flat (np.ndarray): Linear vector of the input problem (1D).
#         k0_uc_orig (float): Constant term of the input problem.
#         kept_indices_uc_relative (list/np.ndarray): Indices (0-based, relative to Q_uc_orig's
#                                                     variables) to keep.
#         elim_indices_uc_relative (list/np.ndarray): Indices (0-based, relative to Q_uc_orig's
#                                                     variables) to eliminate by minimization.
#         tol (float): Tolerance for symmetry checks and rank.
#         rcond_pinv (float): rcond for np.linalg.pinv.

#     Returns:
#         tuple: (Q_red, c_red_flat, k0_red, S_map, t_map_flat)
#             Q_red (np.ndarray): Quadratic matrix for the new (smaller) unconstrained problem.
#             c_red_flat (np.ndarray): Linear vector for this new problem (1D).
#             k0_red (float): Constant term for this new problem.
#             S_map (np.ndarray): Matrix for x_elim_opt = S_map @ x_kept + t_map (n_elim_uc x n_kept_uc).
#             t_map_flat (np.ndarray): Vector for x_elim_opt = S_map @ x_kept + t_map (1D, n_elim_uc).
#     """
#     Q_uc_orig = np.asarray(Q_uc_orig)
#     c_uc_orig_flat = np.asarray(c_uc_orig_flat).flatten()
#     num_vars_uc = Q_uc_orig.shape[0]

#     if c_uc_orig_flat.shape[0] != num_vars_uc:
#         raise ValueError(f"c_uc_orig_flat length must match Q_uc_orig dim.")

#     # Indices are relative to the current unconstrained problem
#     kept_indices_uc_rel, elim_indices_uc_rel, n_k_uc, n_m_uc = _validate_and_partition_indices(
#         num_vars_uc, kept_indices_uc_relative, elim_indices_uc_relative
#     )
#     _check_symmetry(Q_uc_orig, name="Q_uc_orig", tol=tol)

#     # Partition Q_uc_orig and c_uc_orig_flat based on relative kept/elim indices
#     # Pass dummy empty A as it's not used for S,t calculation in this path
#     empty_A_placeholder = np.empty((0, num_vars_uc))
#     # Need to use a general partitioner here as well
#     # Reusing _partition_problem_matrices_for_elim_constraints logic for partitioning Q,c
#     # It takes full Q, c, empty A, then relative kept/elim indices
    
#     Q_kk, Q_km, Q_mk, Q_mm, \
#     c_k_col, c_m_col, \
#     _A_k_dummy, _A_m_dummy = _partition_problem_matrices_for_elim_constraints( 
#         Q_uc_orig, c_uc_orig_flat, empty_A_placeholder, 
#         kept_indices_uc_rel, elim_indices_uc_rel, n_k_uc, n_m_uc
#     )


#     if n_m_uc == 0: # No variables to eliminate by minimization
#         return Q_kk, c_k_col.flatten(), float(k0_uc_orig), np.empty((0,n_k_uc)), np.empty(0)
    
#     Q_mm_pinv = np.linalg.pinv(Q_mm, rcond=rcond_pinv) 

#     rank_Q_mm = np.linalg.matrix_rank(Q_mm, tol=tol)
#     if rank_Q_mm < n_m_uc:
#         # This warning is important: the S_map, t_map give one specific optimal x_elim
#         print(f"Warning: Q_mm in unconstrained reduction was rank-deficient (rank {rank_Q_mm}/{n_m_uc}). "
#               "Pseudo-inverse used; S_map and t_map define a specific (e.g., min norm) optimal x_elim_uc.")

#     # x_m_uc_opt = -Q_mm_pinv @ (Q_mk @ x_k_uc + c_m_col)
#     S_map_final = -Q_mm_pinv @ Q_mk if (Q_mk.size > 0 and Q_mm_pinv.size > 0) else np.zeros((n_m_uc, n_k_uc))
#     t_map_flat_final = -(Q_mm_pinv @ c_m_col).flatten() if (c_m_col.size > 0 and Q_mm_pinv.size > 0) else np.zeros(n_m_uc)
    
#     # Calculate new reduced quadratic parameters using the standard substitution formulas
#     Q_red_final, c_red_final_flat, k0_red_final = _compute_reduced_quadratic_parameters(
#         Q_kk, Q_km, Q_mk, Q_mm, c_k_col, c_m_col, k0_uc_orig, S_map_final, t_map_flat_final
#     )
    
#     return Q_red_final, c_red_final_flat, k0_red_final, S_map_final, t_map_flat_final





### Old ###





def get_constraints_matrix_and_vector(fcns, n):
    """
    Computes the constraint matrix (Jacobian) and constants for a set of constraints.

    Args:
        constraints (list): List of constraint functions.
            n (int): Number of variables.

        Returns:
            tuple: (matrix, constants)
                - matrix: A 2D NumPy array where each row is the vector from a constraint.
                - constants: A 1D NumPy array of constants from each constraint.
    """
    matrix = []
    constants = []

    for fcn in fcns:
        # Compute the constant and vector for the current constraint
        constant = fcn(np.zeros((n, 1)))
        line = fcn(np.identity(n)) - constant

        # Append to the matrix and constants list
        matrix.append(line)
        constants.append(constant)

    # Convert to NumPy arrays
    matrix = np.array(matrix)
    constants = np.array(constants)

    return matrix, constants

def reduce_qp(Q, c, A, b=None, free_indices=None):
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

    if b is None:
      b = np.zeros(A.shape[0])

    Q = np.array(Q, dtype=np.float64)
    c = np.array(c, dtype=np.float64)
    A = np.array(A, dtype=np.float64)

    # Identify indices of free variables
    all_indices = np.arange(Q.shape[0])
    dependent_indices = np.setdiff1d(all_indices, free_indices)

    # Partition matrices and vectors
    Q_dd = Q[np.ix_(dependent_indices, dependent_indices)]
    Q_df = Q[np.ix_(dependent_indices, free_indices)]
    Q_fd = Q[np.ix_(free_indices, dependent_indices)]
    Q_ff = Q[np.ix_(free_indices, free_indices)]

    
    c_f = c[free_indices]
    c_d = c[dependent_indices]

    A_d = A[:, dependent_indices]
    A_f = A[:, free_indices]

    A_d_cond = np.linalg.cond(A_d)
    print("A_d has condition number {}, rank {}, and dimensions {}".format(A_d_cond, np.linalg.matrix_rank(A_d), A_d.shape))
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

    f0_shift = 0.5 * b.T @ A_d_inv.T @ Q_dd @ A_d_inv @ b - c_d.T @ A_d_inv @ b

    # Check if Q_tilde is symmetric
    if np.allclose(Q_tilde, Q_tilde.T):
        print("Q_tilde is symmetric.")
    else:
        print("Q_tilde is not symmetric.")
    # Calculate the Lagrange multiplier
    lambda_ = la.lstsq(A_d.T, c_d - Q_dd @ A_d_inv @ b)[0]
    print("Lagrange multiplier: ", lambda_)
    
    A_d_A_T_inv = la.pinv(A_d @ A_d.T)
    lambda_linear = -la.lstsq(A_d@A_d.T, A_d @ (Q_df - Q_dd @ A_d_inv @ A_f))[0]
    lambda_const = -A_d_A_T_inv @ (A_d @ (Q_dd @ A_d_inv @ b + c_d))

    dep_expr = lambda x_f: A_d_inv_b - A_d_inv_A_f @ x_f

    return Q_tilde, c_tilde, f0_shift, lambda_linear, lambda_const, dep_expr



import numpy as np
import scipy.linalg as la

def lagrange_multiplier_expr(H, f, A, b=None, free_idx=None, rcond=1e-10):
    """
    Compute the linear and constant terms of the Lagrange multipliers λ 
    in terms of free variables (x_f), robust to rank deficiency.

    Args:
        H (ndarray): Symmetric, positive semidefinite Hessian matrix (n x n)
        f (ndarray): Linear term in objective (n,)
        A (ndarray): Constraint matrix (m x n), possibly rank deficient
        b (ndarray): Constraint vector (m,)
        dep_idx (array-like): Indices of dependent variables
        free_idx (array-like): Indices of free variables
        rcond (float): Threshold for singular value cutoff in pseudo-inverse

    Returns:
        W (ndarray): Linear mapping from free variables to multipliers (m x len(x_f))
        w (ndarray): Constant offset for multipliers (m,)
    """
    if b is None:
      b = np.zeros(A.shape[0])
    
    H = np.array(H, dtype=np.float64)
    f = np.array(f, dtype=np.float64)
    A = np.array(A, dtype=np.float64)

    all_indices = np.arange(H.shape[0])
    dep_idx = np.setdiff1d(all_indices, free_idx)


    # Partition A into dependent (Ad) and free (Af) parts
    Ad = A[:, dep_idx]
    Af = A[:, free_idx]

    # Compute pseudoinverse of Ad robustly
    Ad_pinv = la.pinv(Ad, rcond=rcond)

    # Compute Hd, Hf from H
    Hd = H[np.ix_(dep_idx, dep_idx)]
    Hf = H[np.ix_(dep_idx, free_idx)]

    fd  = f[dep_idx]

    m = Ad.shape[0]
    n = Hd.shape[0]
    KKT = np.block([[Hd, Ad.T],
                [Ad, np.zeros((m, m))]])
    #print('KKT', KKT)
    
    #print('KKT_inverse', np.linalg.inv(KKT) )
    #print(f.shape, np.array([1,0,0,0,1, 0]).shape)

    T = 1600
    RT = 8.134*T
    mu0_SiC = -161028
    rho_SiC = 3.21 / 40.11 * 1e6
    rho_Ar = 101e3 / RT

    
    rhs = -np.concatenate([fd, -np.array([rho_SiC,0,0,0,1.*rho_SiC, 0])])
    sol = la.lstsq(KKT, rhs)[0]
    x = sol[:n]
    lambda_ = sol[n:]
    print('sol', x,'lam', lambda_)

    print(dep_idx)

    


    print('Hd shape {}, Hf shape {}'.format(Hd.shape, Hf.shape))

    # Compute intermediate terms for clarity
    # x_d = Ad_pinv @ (b - Af x_f)
    R = Hd @ Ad_pinv @ Af - Hf
    r = Hd @ Ad_pinv @ b + f[dep_idx]

    # Form A transpose robust pseudo-inverse
    Ad_T_pinv = la.pinv(Ad.T, rcond=rcond)

    print('r shape {}, R shape, {}, Ad_t shape {}'.format(r.shape, R.shape, Ad_T_pinv.shape))
    # Linear and constant terms
    W = Ad_T_pinv @ R
    w = -Ad_T_pinv @ r

    return W, w



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
