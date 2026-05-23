import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dimma import train, TrainConfig, NoiseScales, StepInfo


def _per_sample_loss(w, x, y):
    return 0.5 * (jnp.dot(w, x) - y) ** 2


def _make_data(n=8, d=2, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((n, d)).astype(np.float32))
    y = jnp.asarray(rng.standard_normal(n).astype(np.float32))
    return x, y


def _zero_noise():
    return NoiseScales(sigma1=0.0, sigma2=0.0, sigma2_hat=0.0)


def _base_config(T=20, q=5, b1=4, b2=2, seed=0):
    return TrainConfig(
        epsilon=1.0, delta=1e-5, L0=1e6, L1=1e6,
        T=T, q=q, b1=b1, b2=b2, eta=0.01, seed=seed,
    )


def test_smoke_poisson():
    x, y = _make_data()
    cfg = _base_config()
    res = train(x, y, _per_sample_loss, jnp.zeros(2), cfg, _zero_noise(),
                sampler="poisson")
    assert res.params_final is not None
    assert res.params_random is not None
    assert len(res.history.grad_norm) == cfg.T + 1
    assert len(res.history.wall_time_s) == cfg.T + 1
    assert 1 <= res.history.output_step <= cfg.T


def test_smoke_poisson_truncated():
    x, y = _make_data()
    cfg = _base_config()
    res = train(x, y, _per_sample_loss, jnp.zeros(2), cfg, _zero_noise(),
                sampler="poisson_truncated")
    assert len(res.history.grad_norm) == cfg.T + 1


def test_unknown_sampler_raises():
    x, y = _make_data()
    cfg = _base_config()
    with pytest.raises(ValueError) as exc:
        train(x, y, _per_sample_loss, jnp.zeros(2), cfg, _zero_noise(),
              sampler="balls_and_bins")
    msg = str(exc.value)
    assert "poisson" in msg
    assert "poisson_truncated" in msg


def test_callback_fires_once_per_step():
    x, y = _make_data()
    cfg = _base_config()
    seen = []
    train(x, y, _per_sample_loss, jnp.zeros(2), cfg, _zero_noise(),
          step_callback=lambda info: seen.append(info.step))
    assert seen == list(range(cfg.T + 1))


def test_anchor_variation_pattern():
    x, y = _make_data()
    cfg = _base_config(T=20, q=5)
    anchors = []
    train(x, y, _per_sample_loss, jnp.zeros(2), cfg, _zero_noise(),
          step_callback=lambda info: anchors.append((info.step, info.is_anchor)))
    expected = {0, 5, 10, 15, 20}
    actual = {s for s, a in anchors if a}
    assert actual == expected


def test_rng_separation_sampling_independent_of_T():
    x, y = _make_data()

    def run(T):
        cfg = _base_config(T=T)
        masks = []

        def record(info):
            # Use grad_norm changes? Easier: capture the mask via a closure
            pass

        # Indirectly capture by checking that first N grad_norms match
        # across two runs with different T but same seed. Since sampling
        # is privacy-relevant and seed-driven, masks consumed in identical
        # order must produce identical grad_norms (zero noise here).
        hist = []
        train(x, y, _per_sample_loss, jnp.zeros(2), cfg, _zero_noise(),
              step_callback=lambda info: hist.append(info.grad_norm))
        return hist

    h1 = run(15)
    h2 = run(30)
    common = min(len(h1), len(h2))
    for a, b in zip(h1[:common], h2[:common]):
        assert math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)


def test_output_step_in_range():
    x, y = _make_data()
    T = 5
    for seed in range(100):
        cfg = _base_config(T=T, seed=seed)
        res = train(x, y, _per_sample_loss, jnp.zeros(2), cfg, _zero_noise())
        assert 1 <= res.history.output_step <= T


def test_determinism_under_fixed_seed():
    x, y = _make_data()
    cfg = _base_config()
    r1 = train(x, y, _per_sample_loss, jnp.zeros(2), cfg, _zero_noise())
    r2 = train(x, y, _per_sample_loss, jnp.zeros(2), cfg, _zero_noise())
    assert jnp.allclose(r1.params_final, r2.params_final, atol=1e-5)
    assert r1.history.output_step == r2.history.output_step


def test_required_arguments():
    x, y = _make_data()
    cfg = _base_config()
    with pytest.raises(TypeError):
        train(x, y, config=cfg, noise_scales=_zero_noise())  # type: ignore
    with pytest.raises(TypeError):
        train(x, y, _per_sample_loss, config=cfg, noise_scales=_zero_noise())  # type: ignore


def test_delta_w_nan_at_anchor_steps():
    x, y = _make_data()
    cfg = _base_config(T=20, q=5)
    rows = []
    train(x, y, _per_sample_loss, jnp.zeros(2), cfg, _zero_noise(),
          step_callback=lambda info: rows.append((info.is_anchor, info.delta_w)))
    for is_anchor, dw in rows:
        if is_anchor:
            assert math.isnan(dw)
        else:
            assert not math.isnan(dw)
