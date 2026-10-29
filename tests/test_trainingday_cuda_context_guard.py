"""The 4 GiB resource floor must account for CUDA's own initialized context.

This does not imply the model will fit or training will succeed.
"""
import pytest
from trainingday.train_neural import _check_cuda_free_memory

def test_four_gig_gpu_context_overhead_is_not_incorrectly_rejected():
    assert _check_cuda_free_memory(4095.5, 3305.7) == pytest.approx(3295.5)

def test_insufficient_remaining_vram_still_fails_closed():
    with pytest.raises(RuntimeError, match="GPU not sufficiently free"):
        _check_cuda_free_memory(4095.5, 3000.0)

def test_large_gpu_retains_existing_3500_mib_floor():
    assert _check_cuda_free_memory(24000, 3501) == 3500
    with pytest.raises(RuntimeError, match="GPU not sufficiently free"):
        _check_cuda_free_memory(24000, 3499)

def test_under_capacity_remains_rejected():
    with pytest.raises(RuntimeError, match="GPU capacity"):
        _check_cuda_free_memory(3800, 3600)
