import jax
import jax.numpy as jnp
import numpy as np

from dimma.datasets import TabularSplit, arrays_to_split


def test_field_order():
    assert TabularSplit._fields == (
        "x_train", "y_train", "x_test", "y_test", "metadata",
    )


def test_arrays_to_split_cpu():
    x_tr = np.ones((4, 3), dtype=np.float32)
    y_tr = np.zeros(4, dtype=np.float32)
    x_te = np.ones((2, 3), dtype=np.float32)
    y_te = np.zeros(2, dtype=np.float32)
    s = arrays_to_split(x_tr, y_tr, x_te, y_te, device="cpu")
    for arr in (s.x_train, s.y_train, s.x_test, s.y_test):
        assert isinstance(arr, jax.Array)


def test_metadata_default_empty():
    x = np.zeros((2, 2), dtype=np.float32)
    y = np.zeros(2, dtype=np.float32)
    s = arrays_to_split(x, y, x, y, device="cpu")
    assert s.metadata == {}
