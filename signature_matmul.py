import numpy as np
from signature import signatures_via_chen_batch


def level_offsets(dim: int, depth: int):
    offsets = [0]
    for k in range(depth):
        offsets.append(offsets[-1] + dim**k)
    return offsets


def get_level_batch(sig: np.ndarray, offsets, dim: int, level: int) -> np.ndarray:
    start = offsets[level]
    end = start + dim**level
    return sig[:, start:end]


def tensor_exp_batch_matmul(increments: np.ndarray, depth: int) -> np.ndarray:
    """
    Truncated tensor exponential of straight-line increments using matmul.

    Reuses calculations: v^k / k! = (v^(k-1) / (k-1)!) ⊗ (v / k)
    """
    batch_size, _ = increments.shape

    levels = [np.ones((batch_size, 1))]
    current = np.ones((batch_size, 1))

    for k in range(1, depth + 1):
        current = (current[:, :, None] @ increments[:, None, :]).reshape(batch_size, -1) / k
        levels.append(current)

    return np.concatenate(levels, axis=1)


def chen_multiply_batch_matmul(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    dim: int,
    depth: int,
    ) -> np.ndarray:
    """
    Truncated Chen product using levelwise batched matmul instead of np.kron.
    """
    batch_size = sig_a.shape[0]
    offsets = level_offsets(dim, depth)
    out_levels = []

    for n in range(depth + 1):
        level_n = np.zeros((batch_size, dim**n))

        for k in range(n + 1):
            a_k = get_level_batch(sig_a, offsets, dim, k)
            b_nk = get_level_batch(sig_b, offsets, dim, n - k)

            term = (a_k[:, :, None] @ b_nk[:, None, :]).reshape(batch_size, -1)
            level_n += term

        out_levels.append(level_n)

    return np.concatenate(out_levels, axis=1)


def signatures_via_matmul_batch(paths: np.ndarray, depth: int) -> np.ndarray:
    """
    Compute piecewise-linear signatures by:
    1. straight-line tensor exponentials per increment
    2. truncated Chen multiplication via batched matmul
    """
    batch_size, _, dim = paths.shape
    total_dim = sum(dim**k for k in range(depth + 1))

    increments = np.diff(paths, axis=1)

    signatures = np.zeros((batch_size, total_dim))
    signatures[:, 0] = 1.0

    for t in range(increments.shape[1]):
        segment_sig = tensor_exp_batch_matmul(increments[:, t, :], depth)
        signatures = chen_multiply_batch_matmul(signatures, segment_sig, dim, depth)

    return signatures


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    batch_size = 5
    num_times = 20
    dim = 4
    depth = 3

    increments = rng.normal(scale=0.25, size=(batch_size, num_times - 1, dim))
    paths = np.concatenate(
        [np.zeros((batch_size, 1, dim)), np.cumsum(increments, axis=1)],
        axis=1,
    )

    sig_matmul = signatures_via_matmul_batch(paths, depth)
    sig_chen = signatures_via_chen_batch(paths, depth)

    first_matmul = sig_matmul[0]
    first_chen = sig_chen[0]
    diff = first_matmul - first_chen

    print("First path: signature via matmul tensor-exponential / Chen product")
    print(first_matmul)
    print()

    print("First path: signature via reference Chen function")
    print(first_chen)
    print()

    print(f"Error norm = {np.linalg.norm(diff):.6e}")
