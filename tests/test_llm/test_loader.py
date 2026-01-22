"""Tests for LLM loader module."""

import pytest

from omnilex.llm import get_device_info, has_cuda_support, is_kaggle_env


class TestDeviceInfo:
    """Test device info functions."""

    def test_get_device_info_cpu(self):
        """Test device info for CPU mode."""
        assert get_device_info(0) == "CPU"

    def test_get_device_info_gpu_partial(self):
        """Test device info for partial GPU offload."""
        assert get_device_info(10) == "GPU (10 layers offloaded)"

    def test_get_device_info_gpu_all(self):
        """Test device info for full GPU offload."""
        assert get_device_info(-1) == "GPU (all layers offloaded)"


class TestCudaSupport:
    """Test CUDA support detection."""

    def test_has_cuda_support_returns_bool(self):
        """Test that has_cuda_support returns a boolean."""
        result = has_cuda_support()
        assert isinstance(result, bool)


class TestEnvironmentDetection:
    """Test environment detection functions."""

    def test_is_kaggle_env_returns_bool(self):
        """Test that is_kaggle_env returns a boolean."""
        result = is_kaggle_env()
        assert isinstance(result, bool)

    def test_is_kaggle_env_false_locally(self):
        """Test that is_kaggle_env returns False in local environment."""
        # This test runs in CI/local, not Kaggle
        assert is_kaggle_env() is False


class TestLlamaImport:
    """Test llama-cpp-python import."""

    def test_llama_cpp_importable(self):
        """Test that llama-cpp-python can be imported."""
        try:
            from llama_cpp import Llama

            assert Llama is not None
        except ImportError:
            pytest.skip("llama-cpp-python not installed")

    def test_load_model_import_error_without_llama(self):
        """Test that load_model raises ImportError when llama_cpp unavailable."""
        # This test verifies the error handling works
        # In CI, llama_cpp should be installed
        from omnilex.llm.loader import Llama

        if Llama is None:
            from omnilex.llm import load_model

            with pytest.raises(ImportError):
                load_model(model_path="/nonexistent/path")
