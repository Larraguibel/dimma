"""Tests for the reference model + losses migrated into ``dimma.models``.

Covers issue #8 / ADR-0002:
- ``MLP``, ``init_params``, ``forward`` and both BCE losses import from
  ``dimma.models``;
- a forward pass and a per-sample gradient run on a small input;
- the example's ``lib/model.py`` still imports cleanly and re-exports the
  migrated symbols alongside its local pieces.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dimma.models import (
    MLP,
    init_params,
    forward,
    per_sample_bce_loss,
    batch_bce_loss,
    hash_buckets,
    hashed_init_params,
    hashed_forward,
    per_sample_hashed_bce_loss,
)


# ---------------------------------------------------------------------------
# Library imports + forward pass
# ---------------------------------------------------------------------------

def test_symbols_importable_from_dimma_models():
    # MLP is a flax module; the rest are callables.
    assert callable(init_params)
    assert callable(forward)
    assert callable(per_sample_bce_loss)
    assert callable(batch_bce_loss)
    assert issubclass(MLP, object)


def test_forward_single_and_batch_shapes():
    key = jax.random.PRNGKey(0)
    d, hidden = 8, (16, 8)
    params = init_params(key, d, hidden)

    # single example -> scalar logit
    x_single = jnp.ones((d,))
    logit = forward(params, x_single)
    assert logit.shape == ()

    # batch -> (B,)
    x_batch = jnp.ones((5, d))
    logits = forward(params, x_batch)
    assert logits.shape == (5,)


# ---------------------------------------------------------------------------
# Per-sample gradient smoke test (the pattern dimma.train uses)
# ---------------------------------------------------------------------------

def test_per_sample_gradient_smoke():
    key = jax.random.PRNGKey(1)
    d, hidden, B = 8, (16, 8), 4
    params = init_params(key, d, hidden)

    x = jax.random.normal(jax.random.PRNGKey(2), (B, d))
    y = jnp.array([0.0, 1.0, 1.0, 0.0])

    # exactly the contract dimma.train relies on
    per_sample_grad = jax.vmap(
        jax.grad(per_sample_bce_loss), in_axes=(None, 0, 0)
    )
    grads = per_sample_grad(params, x, y)

    # one gradient per example, same pytree structure as params
    leaves = jax.tree.leaves(grads)
    assert leaves, "expected non-empty gradient pytree"
    for leaf in leaves:
        assert leaf.shape[0] == B
        assert jnp.all(jnp.isfinite(leaf))


def test_batch_bce_loss_is_scalar_and_finite():
    key = jax.random.PRNGKey(3)
    d, hidden, B = 8, (16,), 6
    params = init_params(key, d, hidden)
    x = jax.random.normal(jax.random.PRNGKey(4), (B, d))
    y = jnp.zeros((B,))
    loss = batch_bce_loss(params, x, y)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


# ---------------------------------------------------------------------------
# Hashed logistic regression reference model (sparse gradients)
# ---------------------------------------------------------------------------

# Notebook defaults; s = num_fields + num_dense + 1 (bias) = 40.
_HL_BUCKETS = 1024
_HL_FIELDS = 26
_HL_DENSE = 13
_HL_S = _HL_FIELDS + _HL_DENSE + 1


def test_hashed_symbols_importable_and_distinct():
    assert callable(hash_buckets)
    assert callable(hashed_init_params)
    assert callable(hashed_forward)
    assert callable(per_sample_hashed_bce_loss)
    # The hashed model must NOT shadow the MLP's init_params / forward.
    assert hashed_init_params is not init_params
    assert hashed_forward is not forward


def test_hash_buckets_range_and_offset():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 10_000_000, size=(50, _HL_FIELDS))
    out = hash_buckets(ids, _HL_BUCKETS)

    assert out.dtype == np.float32
    assert out.shape == ids.shape
    # Range: every index inside [0, num_fields * num_buckets).
    assert np.all(out >= 0)
    assert np.all(out < _HL_FIELDS * _HL_BUCKETS)
    # Per-column offset: column j occupies block [j*B, (j+1)*B).
    for j in range(_HL_FIELDS):
        col = out[:, j]
        assert np.all(col >= j * _HL_BUCKETS)
        assert np.all(col < (j + 1) * _HL_BUCKETS)
        # Exact mapping id -> j*B + (id % B).
        expected = j * _HL_BUCKETS + (ids[:, j] % _HL_BUCKETS)
        assert np.array_equal(col.astype(np.int64), expected)


def test_hash_buckets_deterministic():
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 1_000_000, size=(20, _HL_FIELDS))
    a = hash_buckets(ids, _HL_BUCKETS)
    b = hash_buckets(ids, _HL_BUCKETS)
    assert np.array_equal(a, b)


def test_hash_buckets_float32_exact_roundtrip():
    # Indices <= 2**24 round-trip exactly through float32.
    rng = np.random.default_rng(2)
    ids = rng.integers(0, 10_000_000, size=(100, _HL_FIELDS))
    out = hash_buckets(ids, _HL_BUCKETS)
    # Cast back to int must equal the exact integer index.
    exact = (np.arange(_HL_FIELDS, dtype=np.int64) * _HL_BUCKETS) + (
        ids.astype(np.int64) % _HL_BUCKETS
    )
    assert np.array_equal(out.astype(np.int64), exact)
    # And the float itself equals the integer (no fractional loss).
    assert np.all(out == exact.astype(np.float32))


def _make_hashed_batch(B=8, seed=0):
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((B, _HL_DENSE)).astype(np.float32)
    raw_ids = rng.integers(0, 5_000_000, size=(B, _HL_FIELDS))
    idx = hash_buckets(raw_ids, _HL_BUCKETS)  # float32 indices
    x = jnp.asarray(np.concatenate([dense, idx], axis=1))
    y = jnp.asarray(rng.integers(0, 2, size=(B,)).astype(np.float32))
    return x, y


def test_hashed_forward_scalar_and_vmap_matches_loop():
    key = jax.random.PRNGKey(0)
    params = hashed_init_params(key, _HL_DENSE, _HL_FIELDS, _HL_BUCKETS)
    x, _ = _make_hashed_batch(B=6, seed=3)

    # single example -> scalar
    logit0 = hashed_forward(params, x[0])
    assert logit0.shape == ()

    # vmap over batch == Python loop
    batched = jax.vmap(hashed_forward, in_axes=(None, 0))(params, x)
    assert batched.shape == (6,)
    looped = jnp.stack([hashed_forward(params, x[i]) for i in range(6)])
    assert jnp.allclose(batched, looped, atol=1e-6)


def test_hashed_gradient_sparsity():
    """Headline assertion: the global per-sample gradient is s-sparse."""
    key = jax.random.PRNGKey(42)
    params = hashed_init_params(key, _HL_DENSE, _HL_FIELDS, _HL_BUCKETS)
    x, y = _make_hashed_batch(B=8, seed=7)

    per_sample_grad = jax.vmap(
        jax.grad(per_sample_hashed_bce_loss), in_axes=(None, 0, 0)
    )
    grads = per_sample_grad(params, x, y)  # pytree with leading batch axis

    table_g = np.asarray(grads["table"])   # (B, num_fields*num_buckets)
    dense_g = np.asarray(grads["w_dense"])  # (B, num_dense)
    bias_g = np.asarray(grads["b"])         # (B,)

    B = x.shape[0]
    max_table_nnz = 0
    max_total_nnz = 0
    for i in range(B):
        table_nnz = int(np.count_nonzero(table_g[i]))
        total_nnz = (
            table_nnz
            + int(np.count_nonzero(dense_g[i]))
            + int(np.count_nonzero(bias_g[i]))
        )
        # Table leaf: at most one nonzero per field.
        assert table_nnz <= _HL_FIELDS, (i, table_nnz)
        # Whole gradient: at most s nonzeros.
        assert total_nnz <= _HL_S, (i, total_nnz)
        max_table_nnz = max(max_table_nnz, table_nnz)
        max_total_nnz = max(max_total_nnz, total_nnz)

    # With random distinct IDs and nonzero dense features / bias, the
    # typical sample hits the maxima.
    assert max_table_nnz == _HL_FIELDS
    assert max_total_nnz == _HL_S


def test_hashed_train_smoke():
    """dimma.train with the hashed model and s-projection returns finite norms."""
    from dimma import train, TrainConfig, NoiseScales

    key = jax.random.PRNGKey(11)
    params = hashed_init_params(key, _HL_DENSE, _HL_FIELDS, _HL_BUCKETS)
    x, y = _make_hashed_batch(B=32, seed=5)

    cfg = TrainConfig(
        epsilon=1.0, delta=1e-5, L0=1e6, L1=1e6,
        T=6, q=3, b1=8, b2=4, eta=0.01, seed=0, s=_HL_S,
    )
    noise = NoiseScales(sigma1=0.0, sigma2=0.0, sigma2_hat=0.0)
    res = train(x, y, per_sample_hashed_bce_loss, params, cfg, noise,
                sampler="poisson")

    assert len(res.history.grad_norm) == cfg.T + 1
    assert all(np.isfinite(g) for g in res.history.grad_norm)


# ---------------------------------------------------------------------------
# The example's lib/model.py still imports cleanly and re-exports the symbols
# ---------------------------------------------------------------------------

def test_example_lib_model_imports_and_reexports():
    lib_root = (
        Path(__file__).parent.parent
        / "examples" / "private_spiderboost" / "lib"
    )
    if str(lib_root) not in sys.path:
        sys.path.insert(0, str(lib_root))

    import model  # noqa: E402  (the example's lib/model.py)

    # migrated symbols are re-exported
    assert model.MLP is MLP
    assert model.init_params is init_params
    assert model.forward is forward
    assert model.per_sample_bce_loss is per_sample_bce_loss
    assert model.batch_bce_loss is batch_bce_loss

    # example-local pieces remain available
    assert callable(model.expected_grad_evals)
    assert callable(model.train_spider)
