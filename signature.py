import numpy as np
from scipy.integrate import solve_ivp


def signatures_via_cde_batch(paths, depth):
    """
    Solve CDEs for multiple paths simultaneously
   
    Args:
        paths: array shape (num_paths, num_times, dim) - batch of paths
        depth: truncation level of signature
       
    Returns:
        signatures: array shape (num_paths, dim_T) - final signatures
    """
    m, T, d = paths.shape
    dim_T = sum(d**k for k in range(depth + 1))
   
    # Precompute tensor level offsets
    offsets = [0]
    for k in range(depth):
        offsets.append(offsets[-1] + d**k)
   
    def cde_rhs(t, S_flat):
        """
        Compute dS/dt for all paths simultaneously.
        S_flat shape: (m * dim_T,)
        """
        # Reshape to (m, dim_T) - each row is one path's signature
        S = S_flat.reshape(m, dim_T)
        dS = np.zeros_like(S)
       
        # Get velocities for all paths at time t
        # Linear interpolation between path points
        idx = min(int(t * (T - 1)), T - 2)
        # Simple difference approximation (as in original code)
        X_dot = (paths[:, idx + 1, :] - paths[:, idx, :]) * (T - 1)  # shape (m, d)
       
        # Compute contribution to each level
        for k in range(depth):
            start_in = offsets[k]
            size = d**k
            start_out = offsets[k + 1]
           
            # Extract level-k signatures for all paths: (m, d^k)
            S_k = S[:, start_in:start_in + size]
           
            # Tensor product S_k ⊗ X_dot -> level k+1
            # Using einsum for batch outer product: (m, d^k) ⊗ (m, d) -> (m, d^{k+1})
            contrib = np.einsum('mi,mj->mij', S_k, X_dot).reshape(m, size * d)
            dS[:, start_out:start_out + size * d] += contrib
       
        return dS.ravel()
   
    # Initial condition: delta at 0 for each path (1 at empty word, 0 elsewhere)
    S0 = np.zeros(m * dim_T)
    S0[::dim_T] = 1.0  # Each path starts with 1 in the scalar component
   
    # Solve the big system
    sol = solve_ivp(
        cde_rhs,
        [0, 1],
        S0,
        method='RK45',
        rtol=1e-10,
        atol=1e-12,
        dense_output=False
    )
   
    # Reshape result to (m, dim_T)
    return sol.y[:, -1].reshape(m, dim_T)


def _level_offsets(dim, depth):
    offsets = [0]
    for k in range(depth):
        offsets.append(offsets[-1] + dim**k)
    return offsets


def _get_level(sig, offsets, dim, level):
    start = offsets[level]
    return sig[start:start + dim**level]


def _chen_multiply(sig_a, sig_b, dim, depth, offsets):
    levels = []
    for n in range(depth + 1):
        level_n = np.zeros(dim**n)
        for k in range(n + 1):
            a_k = _get_level(sig_a, offsets, dim, k)
            b_nk = _get_level(sig_b, offsets, dim, n - k)
            level_n += np.kron(a_k, b_nk)
        levels.append(level_n)
    return np.concatenate(levels)


def _straight_line_signature_batch(increments, depth):
    m, _ = increments.shape
    levels = [np.ones((m, 1))]
    current = np.ones((m, 1))
    for n in range(1, depth + 1):
        current = np.einsum('mi,mj->mij', current, increments).reshape(m, -1) / n
        levels.append(current)
    return np.concatenate(levels, axis=1)


def signatures_via_chen_batch(paths, depth):
    m, T, d = paths.shape
    dim_T = sum(d**k for k in range(depth + 1))
    offsets = _level_offsets(d, depth)
    increments = np.diff(paths, axis=1)

    signatures = np.zeros((m, dim_T))
    signatures[:, 0] = 1.0

    for t in range(T - 1):
        segment_sigs = _straight_line_signature_batch(increments[:, t, :], depth)
        signatures = np.stack([
            _chen_multiply(signatures[i], segment_sigs[i], d, depth, offsets)
            for i in range(m)
        ])

    return signatures


# Example usage with your data
if __name__ == "__main__":
   
    # Create a batch with slight variations
    np.random.seed(41)
    num_paths = 10

    num_times, dim = 30, 4
    paths_batch = np.random.uniform(
        low=-10.0,   # Min value
        high=10.0,    # Max value
        size=(num_paths, num_times, dim)
    )
   
    # Compute signatures for all paths
    depth = 3
    signatures_cde = signatures_via_cde_batch(paths_batch, depth)
    signatures_chen = signatures_via_chen_batch(paths_batch, depth)

    first_cde = signatures_cde[0]
    first_chen = signatures_chen[0]
    first_diff = first_cde - first_chen

    print(f"Computed signatures for {num_paths} paths")
    print(f"Signature dimension: {signatures_cde.shape[1]}")
    print("First path: CDE signature")
    print(first_cde)
    print()
    print("First path: Chen signature")
    print(first_chen)
    print()
    print(f"First path: error norm = {np.linalg.norm(first_diff):.6e}")