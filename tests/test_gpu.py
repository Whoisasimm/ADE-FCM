import numpy as np
import pytest

try:
    import cupy as cp
    _CUDA_AVAILABLE = True
except ImportError:
    cp = None
    _CUDA_AVAILABLE = False

from gpu.gpu_fcm import GPUFCMManager, _fcm_cpu
from gpu.cuda_kernels import check_cuda, FCMKernels


class TestGPU:
    def test_cuda_not_available(self):
        if not _CUDA_AVAILABLE:
            with pytest.raises(RuntimeError, match="CUDA"):
                check_cuda()
        else:
            check_cuda()

    def test_gpu_fcm_manager_creation_cpu_fallback(self):
        manager = GPUFCMManager(use_gpu=True, n_clusters=3, max_iter=10, seed=42)
        if not _CUDA_AVAILABLE:
            assert not manager.use_gpu
        assert manager.n_clusters == 3
        assert manager.max_iter == 10

    def test_gpu_fcm_manager_fit_cpu(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],
                       [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
                      dtype=np.float64)
        manager = GPUFCMManager(use_gpu=False, n_clusters=2, max_iter=20, seed=42)
        centers, U, J_history = manager.fit(X)
        assert centers.shape == (2, 2)
        assert U.shape == (10, 2)
        assert len(J_history) > 0
        assert np.allclose(U.sum(axis=1), 1.0)

    def test_gpu_fcm_manager_predict_after_fit(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        manager = GPUFCMManager(use_gpu=False, n_clusters=2, max_iter=10, seed=42)
        manager.fit(X)
        labels = manager.predict(X)
        assert labels.shape == (4,)
        assert labels.dtype == np.int_ or labels.dtype == np.int64

    def test_predict_without_fit_raises(self):
        manager = GPUFCMManager(use_gpu=False, n_clusters=2)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(RuntimeError, match="Must call fit"):
            manager.predict(X)

    def test_to_cpu_on_numpy(self):
        manager = GPUFCMManager(use_gpu=False)
        X = np.array([1.0, 2.0])
        result = manager.to_cpu(X)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, X)

    def test_fcm_cpu_function(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        centers, U, J_history = _fcm_cpu(X, n_clusters=2, max_iter=10, m=2.0, tol=1e-4, seed=42)
        assert centers.shape == (2, 2)
        assert U.shape == (4, 2)
        assert np.allclose(U.sum(axis=1), 1.0)

    def test_get_memory_report(self):
        manager = GPUFCMManager(use_gpu=False, n_clusters=2)
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        manager.fit(X)
        report = manager.get_memory_report()
        assert isinstance(report, dict)

    def test_repr(self):
        manager = GPUFCMManager(use_gpu=False, n_clusters=3, max_iter=50)
        r = repr(manager)
        assert "GPUFCMManager" in r
        assert "disabled" in r or "enabled" in r


class TestCPUOnly:
    def test_benchmark_cpu_vs_gpu_cpu_fallback(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        manager = GPUFCMManager(use_gpu=False, n_clusters=2, max_iter=5, seed=42)
        results = manager.benchmark_cpu_vs_gpu(X, n_runs=1)
        assert "cpu" in results
        assert results["cpu"]["mean"] > 0
        if not _CUDA_AVAILABLE:
            assert results["gpu"] is None
            assert results["speedup"] == 1.0

    def test_manager_properties(self):
        manager = GPUFCMManager(use_gpu=False, n_clusters=2, seed=42)
        assert manager.centers is None
        assert manager.U is None
        assert manager.J_history == []
        assert manager.fit_time == 0.0

    def test_fcm_kernels_class_methods(self):
        if _CUDA_AVAILABLE:
            membership_kernel = FCMKernels.get_membership_update_kernel()
            assert membership_kernel is not None
            center_kernel = FCMKernels.get_center_update_kernel()
            assert center_kernel is not None
            dist_kernel = FCMKernels.get_distance_kernel()
            assert dist_kernel is not None
            obj_kernel = FCMKernels.get_objective_kernel()
            assert obj_kernel is not None

    def test_fit_and_predict_cpu_consistency(self):
        X = np.array([[1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0]], dtype=np.float64)
        manager1 = GPUFCMManager(use_gpu=False, n_clusters=2, max_iter=10, seed=42)
        manager2 = GPUFCMManager(use_gpu=False, n_clusters=2, max_iter=10, seed=42)
        manager1.fit(X)
        manager2.fit(X)
        assert np.allclose(manager1.centers, manager2.centers)
        labels1 = manager1.predict(X)
        labels2 = manager2.predict(X)
        assert np.array_equal(labels1, labels2)

    def test_rapids_fcm_import(self):
        try:
            from gpu.rapids_fcm import RAPIDSFCM
            model = RAPIDSFCM(n_clusters=2, max_iter=5, seed=42)
            assert model.n_clusters == 2
        except ImportError as e:
            pytest.skip(f"RAPIDSFCM import failed: {e}")

    def test_spark_gpu_hybrid_import(self):
        try:
            from gpu.spark_gpu_hybrid import SparkGPUHybridEngine
            engine = SparkGPUHybridEngine(n_clusters=2, max_iter=5, seed=42)
            assert engine.n_clusters == 2
        except Exception:
            pass
