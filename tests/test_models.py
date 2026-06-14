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

from dimma.models import (
    MLP,
    init_params,
    forward,
    per_sample_bce_loss,
    batch_bce_loss,
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
