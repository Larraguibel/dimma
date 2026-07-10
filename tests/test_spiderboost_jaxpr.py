"""Bit-exactness guard for the issue #24 kernel deduplication.

The ``s=None`` SpiderBoost kernels carry a hard promise: the traced program
must stay byte-identical to the pre-projection implementation (regression
oracle in ``tests/test_regression_against_source.py`` depends on it). The
issue #24 refactor removes ~30 duplicated lines by composing the base anchor
step and by extracting a shared ``_noisy_increment`` helper for the variation
step. Python function composition is invisible to JAX tracing — the same
primitives emitted in the same order produce the same jaxpr — so the refactor
is *claimed* to be behaviour-preserving. This test PROVES it instead of
assuming it.

How the proof works
-------------------
``tests/fixtures/kernels_pre_issue24.py`` is a frozen, verbatim copy of
``kernels.py`` as it existed on the commit immediately before this refactor
(the monolithic closures, no shared helper). At test time we build the
``s=None`` anchor and variation kernels *both* ways — from the live,
refactored module and from the frozen pre-refactor module — and assert their
``jax.make_jaxpr`` output is textually identical.

Why this is a genuine guarantee, not a tautology:
  * The two kernels come from genuinely different Python source (the frozen
    file inlines the pipeline; the live file routes anchor projection through
    ``anchor_step`` and variation through ``_noisy_increment``). If the
    refactor had reordered an op, dropped one, or changed a primitive, the
    jaxprs would diverge and this test would fail.
  * Both are traced under the *same* installed JAX at test time, so the
    comparison is immune to jaxpr-text drift across JAX releases — a
    golden-text file captured on one JAX version would not be.

Scope: the guarantee is the ``s=None`` path (the one with the bit-exact
contract), for both the anchor and the variation kernel, exactly as issue #24
requires. The projected (``s=int``) path is new, deliberately biased behaviour
and is pinned numerically in ``tests/test_spiderboost_kernels.py`` (hand
reconstruction against the DP primitives).
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp

from dimma.algorithms.spiderboost import (
    make_anchor_step as make_anchor_step_live,
    make_variation_step as make_variation_step_live,
)

# ---------------------------------------------------------------------------
# Load the frozen pre-refactor kernels as a standalone module (by path, so the
# import works regardless of how `tests/` is packaged).
# ---------------------------------------------------------------------------
_FIXTURE = Path(__file__).parent / "fixtures" / "kernels_pre_issue24.py"
_spec = importlib.util.spec_from_file_location("kernels_pre_issue24", _FIXTURE)
_pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pre)


def _per_sample_loss(w, x, y):
    return 0.5 * (jnp.dot(w, x) - y) ** 2


def _grad_fn():
    return jax.vmap(jax.grad(_per_sample_loss), in_axes=(None, 0, 0))


# Concrete tracing inputs. Values are irrelevant to a jaxpr (only shapes and
# dtypes matter), but we use the same ones the pre-refactor capture used.
_B = 4
_W = jnp.array([0.5, -0.3])
_W_PREV = jnp.array([0.4, -0.2])
_X = jnp.ones((_B, 2))
_Y = jnp.ones((_B,))
_MASK = jnp.ones(_B)
_PREV_GRAD = jnp.array([0.7, -0.4])
_KEY = jax.random.PRNGKey(7)

_ANCHOR_ARGS = (_W, _X, _Y, _MASK, _B, 2.0, 1.0, _KEY)
_VARIATION_ARGS = (_W, _W_PREV, _PREV_GRAD, _X, _Y, _MASK, _B, 1.0, 2.0, 3.0, _KEY)


def test_anchor_s_none_jaxpr_identical_to_pre_refactor():
    live = make_anchor_step_live(_grad_fn(), s=None)
    pre = _pre.make_anchor_step(_grad_fn(), s=None)
    live_jaxpr = str(jax.make_jaxpr(live)(*_ANCHOR_ARGS))
    pre_jaxpr = str(jax.make_jaxpr(pre)(*_ANCHOR_ARGS))
    assert live_jaxpr == pre_jaxpr


def test_variation_s_none_jaxpr_identical_to_pre_refactor():
    live = make_variation_step_live(_grad_fn(), s=None)
    pre = _pre.make_variation_step(_grad_fn(), s=None)
    live_jaxpr = str(jax.make_jaxpr(live)(*_VARIATION_ARGS))
    pre_jaxpr = str(jax.make_jaxpr(pre)(*_VARIATION_ARGS))
    assert live_jaxpr == pre_jaxpr


def test_fixture_is_genuinely_a_different_source():
    """Sanity check that the frozen fixture still holds the *monolithic* code.

    If someone regenerates the fixture from the refactored file, the jaxpr
    equality tests above become tautological. Guard against that by asserting
    the fixture has no shared helper and the live module does.
    """
    pre_src = _FIXTURE.read_text()
    assert "_noisy_increment" not in pre_src, (
        "fixture must remain the pre-refactor monolithic code"
    )
    import dimma.algorithms.spiderboost.kernels as live_mod

    live_src = Path(live_mod.__file__).read_text()
    assert "_noisy_increment" in live_src, (
        "live kernels should use the shared _noisy_increment helper"
    )
