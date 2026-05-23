import pytest
import jax

from dimma.utils import resolve_device


def test_cpu_resolves():
    d = resolve_device("cpu")
    assert isinstance(d, jax.Device)


def test_case_insensitive():
    assert resolve_device("CPU") == resolve_device("cpu")


def test_cuda_gpu_alias():
    try:
        gpu_devs = jax.devices("gpu")
    except RuntimeError:
        pytest.skip("No GPU backend available")
    if not gpu_devs:
        pytest.skip("No GPU devices")
    assert resolve_device("cuda") == resolve_device("gpu")


def test_unknown_raises():
    with pytest.raises(ValueError):
        resolve_device("xpu")
