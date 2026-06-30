import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.spatial.distance import cdist as gpu_cdist
    _CUDA_AVAILABLE = True
except ImportError:
    cp = None
    _CUDA_AVAILABLE = False


def check_cuda():
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA/CuPy not available. Install cupy: pip install cupy")


def _get_module():
    check_cuda()
    return cp


class FCMKernels:
    _mem_kernel = None
    _center_kernel = None
    _dist_kernel = None
    _obj_kernel = None

    @classmethod
    def get_membership_update_kernel(cls):
        if cls._mem_kernel is not None:
            return cls._mem_kernel
        cp = _get_module()
        cls._mem_kernel = cp.ElementwiseKernel(
            "float64 dist, float64 inv_exponent",
            "float64 membership",
            "membership = pow(dist, inv_exponent)",
            "membership_update",
        )
        return cls._mem_kernel

    @classmethod
    def get_center_update_kernel(cls):
        if cls._center_kernel is not None:
            return cls._center_kernel
        cp = _get_module()
        cls._center_kernel = cp.RawKernel(r"""
        extern "C" __global__
        void center_update(
            const double* data, const double* U, double* new_centers,
            int n_samples, int n_features, int n_clusters
        ) {
            int c = blockIdx.x;
            int f = threadIdx.x;
            if (c >= n_clusters || f >= n_features) return;

            double num = 0.0;
            double den = 0.0;
            for (int i = 0; i < n_samples; i++) {
                double u = U[i * n_clusters + c];
                num += u * data[i * n_features + f];
                den += u;
            }
            new_centers[c * n_features + f] = (den > 1e-15) ? num / den : 0.0;
        }
        """, "center_update")
        return cls._center_kernel

    @classmethod
    def get_distance_kernel(cls):
        if cls._dist_kernel is not None:
            return cls._dist_kernel
        cp = _get_module()
        cls._dist_kernel = cp.ElementwiseKernel(
            "float64 x, float64 y",
            "float64 dist",
            "dist = (x - y) * (x - y)",
            "distance_computation",
        )
        return cls._dist_kernel

    @classmethod
    def get_objective_kernel(cls):
        if cls._obj_kernel is not None:
            return cls._obj_kernel
        cp = _get_module()
        cls._obj_kernel = cp.ReductionKernel(
            "float64 val",
            "float64 obj",
            "val",
            "a + b",
            "obj = 0.0",
            "objective_function",
        )
        return cls._obj_kernel


def compute_membership_gpu(distances, m):
    cp = _get_module()
    exponent = -2.0 / (m - 1.0)

    kernel = FCMKernels.get_membership_update_kernel()
    inv_dist = kernel(distances, exponent)

    row_sums = cp.sum(inv_dist, axis=1, keepdims=True)
    return inv_dist / cp.maximum(row_sums, 1e-15)


def compute_centers_gpu(data, U, fuzziness):
    cp = _get_module()
    n_samples, n_features = data.shape
    n_clusters = U.shape[1]

    Um = U ** fuzziness
    new_centers = cp.zeros((n_clusters, n_features), dtype=cp.float64)

    kernel = FCMKernels.get_center_update_kernel()
    block = (n_features,)
    grid = (n_clusters,)
    kernel(grid, block, (data, Um, new_centers, n_samples, n_features, n_clusters))

    return new_centers


def compute_distances_gpu(data, centers):
    cp = _get_module()
    return gpu_cdist(data, centers, metric="sqeuclidean")


def compute_objective_gpu(U, distances, m):
    cp = _get_module()
    kernel = FCMKernels.get_objective_kernel()
    val = (U ** m) * distances
    return float(kernel(val))
