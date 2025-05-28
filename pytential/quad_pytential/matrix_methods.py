import numpy as np
import scipy.linalg as la
from scipy.linalg import qr, solve_triangular

DEFAULT_TOL = 1e-8
RCOND_PINV = 1e-12

def _check_symmetry(Q, name="Q_matrix", tol=DEFAULT_TOL):
    """Checks if a matrix Q is square and symmetric."""
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix. Got shape {Q.shape}")
    if not np.allclose(Q, Q.T, atol=tol, rtol=tol):
        raise ValueError(f"{name} must be a symmetric matrix.")

def _validate_and_derive_indices(num_total_vars, priority_kept_indices):
    """Validates priority_kept_indices and derives elim_candidate_indices."""
    # priority_kept_indices = np.array(sorted(list(set(priority_kept_indices))), dtype=int) # Sort and unique
    #priority_kept_indices = np.array(list(set(priority_kept_indices)), dtype=int) # Unique, but NO SORT
    priority_kept_indices = np.asarray(priority_kept_indices, dtype=int)
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

    assert n_kept + n_elim_cand == num_total_vars, "Index partitioning failed."

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

def _compute_reduced_quadratic_parameters(
    Q_kk, Q_km, Q_mk, Q_mm, 
    c_k_col, c_m_col, 
    k0_full, 
    S, t_flat):
    """
    Calculates the coefficients (Q_red, c_red_flat, k0_red) of the new quadratic 
    function in terms of x_k, after substituting x_m = S @ x_k + t.

    Args:
        Q_kk (np.ndarray): Quadratic block for kept variables (n_k x n_k).
        Q_km (np.ndarray): Quadratic cross-term block (n_k x n_m).
        Q_mk (np.ndarray): Quadratic cross-term block (n_m x n_k), Q_km.T.
        Q_mm (np.ndarray): Quadratic block for eliminated variables (n_m x n_m).
        c_k_col (np.ndarray): Linear part for kept variables (n_k x 1).
        c_m_col (np.ndarray): Linear part for eliminated variables (n_m x 1).
        k0_full (float): Original constant term.
        S (np.ndarray): Transformation matrix (n_m x n_k) from x_m = S @ x_k + t.
        t_flat (np.ndarray): Transformation vector (1D, n_m) from x_m = S @ x_k + t.

    Returns:
        tuple: (Q_red, c_red_flat, k0_red)
            Q_red (np.ndarray): New quadratic matrix for x_k.
            c_red_flat (np.ndarray): New linear vector for x_k (1D).
            k0_red (float): New constant term.
    """
    # Ensure t is a column vector for matrix operations
    t_col = np.asarray(t_flat).reshape(-1, 1)

    # Get dimensions
    n_k = Q_kk.shape[0]
    n_m = Q_mm.shape[0] # S.shape[0] or t_flat.shape[0] are also n_m

    # --- Initialize with parts purely dependent on x_k ---
    Q_red = Q_kk.copy() if Q_kk.size > 0 else np.zeros((n_k, n_k))
    c_red = c_k_col.copy() if c_k_col.size > 0 else np.zeros((n_k, 1))
    k0_red = float(k0_full)

    # --- Add terms arising from the substitution x_m = S @ x_k + t ---
    # These terms only contribute if n_m > 0 (i.e., if there were x_m variables)
    if n_m > 0:
        # --- Q_red terms ---
        # Q_red = Q_kk + Q_km@S + S.T@Q_mk + S.T@Q_mm@S
        if n_k > 0: # Cross terms with x_k only exist if n_k > 0
            if Q_km.size > 0 and S.size > 0:  # Q_km is (n_k x n_m), S is (n_m x n_k)
                Q_red += Q_km @ S
            if Q_mk.size > 0 and S.size > 0:  # Q_mk is (n_m x n_k), S.T is (n_k x n_m)
                Q_red += S.T @ Q_mk
        
        # S.T @ Q_mm @ S is (n_k x n_m) @ (n_m x n_m) @ (n_m x n_k) -> (n_k x n_k)
        # This term is valid even if n_k=0 (S would be n_m x 0, Q_red is 0x0)
        if S.size > 0 and Q_mm.size > 0: # Check S might be empty if n_k=0
             if S.shape[1] == Q_red.shape[1]: # Ensure S is compatible with Q_red dims
                Q_red += S.T @ Q_mm @ S

        # --- c_red terms ---
        # c_red = c_k + Q_km@t + S.T@Q_mm@t + S.T@c_m
        if n_k > 0: # Q_km @ t results in (n_k x 1)
            if Q_km.size > 0 and t_col.size > 0:
                c_red += Q_km @ t_col
        
        # S.T @ Q_mm @ t results in (n_k x 1)
        # S.T @ c_m results in (n_k x 1)
        if S.size > 0: # S is (n_m x n_k), S.T is (n_k x n_m)
            if Q_mm.size > 0 and t_col.size > 0:
                c_red += S.T @ Q_mm @ t_col
            if c_m_col.size > 0:
                c_red += S.T @ c_m_col
        
        # --- k0_red terms ---
        # k0_red = k0_full + c_m.T@t + 0.5 * t.T@Q_mm@t
        if c_m_col.size > 0 and t_col.size > 0:
            k0_red += (c_m_col.T @ t_col).item()
        if Q_mm.size > 0 and t_col.size > 0:
            k0_red += (0.5 * t_col.T @ Q_mm @ t_col).item()
            
    return Q_red, c_red.flatten(), k0_red

def _partition_qc_for_unconstrained_reduction(
    Q_orig, c_orig_flat, 
    user_kept_indices_rel 
    ):
    """
    Partitions Q and c based on relative kept and (derived) eliminated variable 
    indices for an unconstrained quadratic problem reduction.

    Args:
        Q_orig (np.ndarray): The quadratic matrix of the current unconstrained problem.
        c_orig_flat (np.ndarray): The linear vector (1D) of the current problem.
        user_kept_indices_rel (list/np.ndarray): 0-based relative indices of variables to keep.
                                                  Other variables will be considered for elimination.
    Returns:
        tuple: (Q_kk, Q_km, Q_mk, Q_mm, c_k_col, c_m_col, n_k, n_m)
               n_k and n_m are the number of kept and eliminated variables.
    """
    num_vars = Q_orig.shape[0]
    if c_orig_flat.shape[0] != num_vars:
        raise ValueError(f"c_orig_flat length ({c_orig_flat.shape[0]}) must match Q_orig dim ({num_vars}).")

   
    # Derive elimination indices and get counts
    # _validate_and_derive_indices takes (total_vars, kept_to_validate_and_prioritize)
    # and returns (validated_kept, derived_elim, count_kept, count_elim)
    kept_indices, elim_indices, n_k, n_m = _validate_and_derive_indices(
        num_vars, user_kept_indices_rel
    )
   
    # Partition Q
    Q_kk = Q_orig[np.ix_(kept_indices, kept_indices)] if n_k > 0 else np.empty((0, 0))
 
    if n_m > 0:
        Q_mm = Q_orig[np.ix_(elim_indices, elim_indices)]
        if n_k > 0:
            Q_km = Q_orig[np.ix_(kept_indices, elim_indices)]
            Q_mk = Q_orig[np.ix_(elim_indices, kept_indices)] # Typically Q_km.T
        else: # n_k == 0
            Q_km = np.empty((0, n_m))
            Q_mk = np.empty((n_m, 0))
    else: # n_m == 0 (all variables are kept)
        Q_mm = np.empty((0, 0))
        Q_km = np.empty((n_k, 0)) if n_k > 0 else np.empty((0,0))
        Q_mk = np.empty((0, n_k)) if n_k > 0 else np.empty((0,0))

    # Partition c (ensuring column vectors)
    c_k_col = c_orig_flat[kept_indices].reshape(-1, 1) if n_k > 0 else np.empty((0, 1))
    c_m_col = c_orig_flat[elim_indices].reshape(-1, 1) if n_m > 0 else np.empty((0, 1))
        
    return Q_kk, Q_km, Q_mk, Q_mm, c_k_col, c_m_col, n_k, n_m


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

def reduce_unconstrained_quadratic_by_minimization(
    Q_uc_orig, c_uc_orig_flat, k0_uc_orig, 
    kept_indices_uc_relative, 
    tol=DEFAULT_TOL, rcond_pinv=RCOND_PINV):
    """
    Reduces dimensionality of an UNCONSTRAINED quadratic problem by minimizing out
    variables not in kept_indices_uc_relative.
    Q_uc_orig must be positive semidefinite.
    """
    Q_uc_orig = np.asarray(Q_uc_orig)
    c_uc_orig_flat = np.asarray(c_uc_orig_flat).flatten()
    num_vars_uc = Q_uc_orig.shape[0] # This is now the primary source for num_vars_uc

    # Initial check on c_uc_orig_flat length against Q_uc_orig
    if c_uc_orig_flat.shape[0] != num_vars_uc:
        raise ValueError(f"c_uc_orig_flat length ({c_uc_orig_flat.shape[0]}) "
                         f"must match Q_uc_orig dim ({num_vars_uc}).")

    _check_symmetry(Q_uc_orig, name="Q_uc_orig", tol=tol)

    # Partition Q_uc_orig and c_uc_orig_flat. 
    # _partition_qc_for_unconstrained_reduction will internally validate kept_indices_uc_relative
    # and derive elim_indices, n_k_uc, n_m_uc.


    Q_kk, Q_km, Q_mk, Q_mm, \
    c_k_col, c_m_col, \
    n_k_uc, n_m_uc = _partition_qc_for_unconstrained_reduction( 
        Q_uc_orig, c_uc_orig_flat, 
        kept_indices_uc_relative
    )

    
    if n_m_uc == 0: # No variables to eliminate by minimization
        return Q_kk, c_k_col.flatten(), float(k0_uc_orig), np.empty((0,n_k_uc)), np.empty(0)
    
    Q_mm_pinv = np.linalg.pinv(Q_mm, rcond=rcond_pinv)

    rank_Q_mm = np.linalg.matrix_rank(Q_mm, tol=tol)
    if rank_Q_mm < n_m_uc:
        print(f"Warning: Q_mm in unconstrained reduction was rank-deficient (rank {rank_Q_mm}/{n_m_uc}). "
              "Pseudo-inverse used; S_map and t_map define a specific (e.g., min norm) optimal x_elim_uc.")

    S_map_final = np.zeros((n_m_uc, n_k_uc))
    if Q_mk.size > 0 and Q_mm_pinv.size > 0 and n_k_uc > 0: # Check n_k_uc for Q_mk relevance
        S_map_final = -Q_mm_pinv @ Q_mk
    
    t_map_flat_final = np.zeros(n_m_uc)
    if c_m_col.size > 0 and Q_mm_pinv.size > 0:
        t_map_flat_final = -(Q_mm_pinv @ c_m_col).flatten()
    
    Q_red_final, c_red_final_flat, k0_red_final = _compute_reduced_quadratic_parameters(
        Q_kk, Q_km, Q_mk, Q_mm, c_k_col, c_m_col, k0_uc_orig, S_map_final, t_map_flat_final
    )
    
    return Q_red_final, c_red_final_flat, k0_red_final, kept_indices_uc_relative, S_map_final, t_map_flat_final


