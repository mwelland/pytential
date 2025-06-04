import numpy as np
import scipy.linalg as la
from scipy.linalg import qr, solve_triangular

import numpy as np

def _clip_small_elements(matrix, clip_threshold_percent):
    """Clips elements smaller than a threshold in a matrix."""
    avg = np.mean(np.abs(matrix))
    threshold = clip_threshold_percent * avg
    matrix[np.abs(matrix) < threshold] = 0
    return matrix

def get_partial_convex_conjugate_quad_params(H_orig, b_orig, c_orig, conjugate_indices, clip_threshold_percent=1e-6):
    """
    Calculates parameters for the partial convex conjugate of a quadratic function.

    Original: f(x) = 0.5 * x.T @ H_orig @ x + b_orig.T @ x + c_orig.
    Transformed: f*(v) = 0.5 * v.T @ M @ v + N.T @ v + P, where v = (y_a, x_b).
    y_a are conjugate variables for x_a (from conjugate_indices), x_b are other variables.
    The transformed quadratic is defined on the affine subspace A_cond @ v = c_cond.
    Outside this subspace, f*(v) = +infinity.

    Note that the variable order is shifted as part of this operation. The mapping is given
    by the ordered_conj_indices and ordered_param_indices.

    Args:
        H_orig (np.ndarray): Hessian matrix of the original quadratic.
        b_orig (np.ndarray): Linear term vector of the original quadratic (n, or n,1).
        c_orig (float): Constant term of the original quadratic.
        conjugate_indices (list or tuple of int): 0-based indices of variables
                                                   to take the conjugate with respect to.

    Returns:
        tuple: (M, N, P, A_cond, c_cond, ordered_conj_indices, ordered_param_indices)
            M (np.ndarray): Hessian of the transformed quadratic.
            N (np.ndarray): Linear term of the transformed quadratic (column vector).
            P (float): Constant term of the transformed quadratic.
            A_cond (np.ndarray): Matrix for the finiteness condition.
            c_cond (np.ndarray): Vector for the finiteness condition (column vector).
            ordered_conj_indices (list): Sorted original indices for y_a variables.
            ordered_param_indices (list): Sorted original indices for x_b variables.
                                         Variables in M, N, A_cond are ordered as (y_a, x_b).
    """
    H = np.asarray(H_orig, dtype=float)
    b_col = np.asarray(b_orig, dtype=float).reshape(-1, 1) # Ensure b is a column vector
    c = float(c_orig)
    n_total = H.shape[0]

    if H.shape != (n_total, n_total) or \
       (b_col.shape[0] != n_total and n_total != 0) or \
       (b_col.shape[0] == n_total and b_col.shape[1] != 1): # Allow b to be (0,1) if n_total=0
        raise ValueError(
            f"Dimension mismatch. H must be ({n_total},{n_total}), b must be ({n_total},1) or ({n_total},)."
        )

    # Handle n_total = 0 case explicitly for clarity, though numpy might handle some parts.
    if n_total == 0:
        ordered_conj_idx = sorted(list(set(int(i) for i in conjugate_indices)))
        ordered_param_idx = [i for i in range(0) if i not in ordered_conj_idx] # Will be empty
        if ordered_conj_idx: # Should not happen if n_total is 0
             raise ValueError("Conjugate indices provided for 0-dimensional quadratic.")
        return (np.zeros((0,0)), np.zeros((0,1)), -c, 
                np.zeros((0,0)), np.zeros((0,1)), 
                ordered_conj_idx, ordered_param_idx)

    conj_idx_orig = sorted(list(set(int(i) for i in conjugate_indices)))
    if any(i < 0 or i >= n_total for i in conj_idx_orig):
        raise ValueError("Conjugate indices out of bounds.")
        
    all_original_indices = list(range(n_total))
    param_idx_orig = sorted([i for i in all_original_indices if i not in conj_idx_orig])

    perm_order = conj_idx_orig + param_idx_orig
    
    H_perm = H[np.ix_(perm_order, perm_order)]
    b_perm = b_col[perm_order]

    num_conj = len(conj_idx_orig)
    # num_param = len(param_idx_orig) # not explicitly used later other than for H_ab etc.

    H_aa = H_perm[:num_conj, :num_conj]
    H_ab = H_perm[:num_conj, num_conj:]
    H_bb = H_perm[num_conj:, num_conj:]
    b_a = b_perm[:num_conj]
    b_b = b_perm[num_conj:]

    K = np.linalg.pinv(H_aa) # K is num_conj x num_conj

    # Transformed Hessian M
    M_aa_t = K
    M_ab_t = -K @ H_ab
    M_ba_t = -H_ab.T @ K 
    M_bb_t = (H_ab.T @ K @ H_ab) - H_bb
    
    # np.block handles cases where some blocks are 0-dimensional (e.g., num_conj=0 or num_param=0)
    M_transformed = np.block([[M_aa_t, M_ab_t], [M_ba_t, M_bb_t]])

    # Transformed linear term N
    N_a_t = -K @ b_a
    N_b_t = (H_ab.T @ K @ b_a) - b_b
    
    # np.vstack handles cases where one part is 0-dimensional
    N_transformed = np.vstack([N_a_t, N_b_t])
        
    # Transformed constant P
    # (b_a.T @ K @ b_a) results in scalar 0.0 if num_conj=0, or 1x1 matrix if num_conj>0
    val_bkb_term = b_a.T @ K @ b_a
    val_bkb = val_bkb_term.item() if num_conj > 0 else float(val_bkb_term) 
    P_transformed = 0.5 * val_bkb - c

    # Finiteness condition: A_cond @ v = c_cond
    # L_cond_matrix @ y_a - L_cond_matrix @ H_ab @ x_b = L_cond_matrix @ b_a
    if num_conj > 0:
        L_cond_matrix = np.eye(num_conj) - (H_aa @ K)
    else: # num_conj == 0, L_cond_matrix is 0x0.
        L_cond_matrix = np.zeros((0, 0)) 

    A_cond_ya_part = L_cond_matrix
    A_cond_xb_part = -L_cond_matrix @ H_ab # Results in (num_conj x num_param)
    
    A_condition = np.block([[A_cond_ya_part, A_cond_xb_part]])
    c_condition = L_cond_matrix @ b_a

    M_transformed = _clip_small_elements(M_transformed, clip_threshold_percent)
    N_transformed = _clip_small_elements(N_transformed, clip_threshold_percent)
    A_condition = _clip_small_elements(A_condition, clip_threshold_percent)

    # M_avg = np.mean(np.abs(M_transformed))
    # M_threshold = clip_threshold_percent * M_avg
    # M_transformed[np.abs(M_transformed) < M_threshold] = 0

    return (M_transformed, N_transformed, P_transformed, 
            A_condition, c_condition, 
            conj_idx_orig, param_idx_orig)

