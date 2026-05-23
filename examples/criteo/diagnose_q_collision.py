"""Diagnose the q=100 vs q=200 AUC collision observed in Phase 5.

Phase 5's phase_length_q_tradeoff notebook reported that AUC(w_bar)
at q=100 and q=200 both came out to 0.6094 (to 4 decimal places).
This script runs the two configurations side by side with maximum
precision and prints enough diagnostic information to distinguish:

1. Benign coincidence at 4 decimals (different output_step, different
   params_random, AUC differs in higher decimals).
2. Same output_step, different params_random, AUC rounds the same.
3. Suspicious: same output_step, identical params_random.
4. Very suspicious: different params_random, identical AUC to many
   decimals.

The script prints the data; interpretation is manual.

Usage
-----
    cd dimma/examples/criteo
    python3 diagnose_q_collision.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import roc_auc_score

from dimma import TrainConfig, compute_noise_scales, train
from dimma.datasets import load_criteo

import model


# Hyperparameters match phase_length_q_tradeoff.ipynb (Phase 5) exactly.
SEED = 0
T = 200
EPSILON = 10.0
B1 = 8192
B2 = 512
L0 = 3.0
L1 = 5.0
ETA = 0.01
HIDDEN_DIMS = (64, 32)
Q_VALUES = [100, 200]


def make_evaluate(forward_jit):
    def evaluate_auc(params, x, y):
        logits = forward_jit(params, x)
        return float(roc_auc_score(np.asarray(y), np.asarray(logits)))
    return evaluate_auc


def pytree_max_abs_diff(a, b) -> float:
    leaves_a = jax.tree.leaves(a)
    leaves_b = jax.tree.leaves(b)
    return max(
        float(jnp.max(jnp.abs(la - lb))) for la, lb in zip(leaves_a, leaves_b)
    )


def pytree_first_leaf_head(pytree, n_values: int = 3) -> np.ndarray:
    first_leaf = jax.tree.leaves(pytree)[0]
    return np.asarray(first_leaf.flatten()[:n_values])


def main() -> None:
    print("=" * 72)
    print("Diagnostic: q=100 vs q=200 AUC collision")
    print("=" * 72)

    print("Loading Criteo (integer mode)...")
    split = load_criteo(features="integer", seed=SEED)
    n_train = int(split.x_train.shape[0])
    input_dim = int(split.x_train.shape[1])
    # Notebook uses delta=1/(1.1*n_train); apply the same here.
    delta = 1.0 / (1.1 * n_train)

    print(f"Seed: {SEED}, T: {T}, epsilon: {EPSILON}, delta: {delta:.4e}")
    print(f"b1: {B1}, b2: {B2}, L0: {L0}, L1: {L1}, eta: {ETA}")
    print(f"hidden_dims: {HIDDEN_DIMS}")
    print(f"n_train: {n_train}, n_test: {int(split.x_test.shape[0])}, "
          f"input_dim: {input_dim}")
    print()

    # Identical initial parameters for all runs (matches notebook 2).
    key = jax.random.PRNGKey(SEED)
    params_init = model.init_params(
        key, input_dim=input_dim, hidden_dims=HIDDEN_DIMS
    )

    forward_jit = jax.jit(model.forward)
    evaluate_auc = make_evaluate(forward_jit)

    results = {}
    for q in Q_VALUES:
        print(f"Running q={q}...")
        config = TrainConfig(
            epsilon=EPSILON, delta=delta,
            L0=L0, L1=L1,
            T=T, q=q, b1=B1, b2=B2,
            eta=ETA, seed=SEED,
        )
        noise_scales = compute_noise_scales(
            L0=L0, L1=L1, epsilon=EPSILON, delta=delta,
            T=T, q=q, n=n_train, b1=B1, b2=B2,
        )
        result = train(
            split.x_train, split.y_train,
            per_sample_loss_fn=model.per_sample_bce_loss,
            init_params=params_init,
            config=config,
            noise_scales=noise_scales,
        )
        results[q] = result
        print(f"  done. output_step={result.history.output_step}, "
              f"wall_time_sum={sum(result.history.wall_time_s):.2f}s")
    print()

    print("-" * 72)
    print("Per-run diagnostics")
    print("-" * 72)
    for q in Q_VALUES:
        r = results[q]
        auc_rand = evaluate_auc(r.params_random, split.x_test, split.y_test)
        auc_final = evaluate_auc(r.params_final, split.x_test, split.y_test)
        gn = r.history.grad_norm
        os_idx = r.history.output_step
        print(f"q={q}")
        print(f"  output_step (t*):           {os_idx}")
        print(f"  AUC(w_bar = params_random): {auc_rand:.10f}")
        print(f"  AUC(w_T   = params_final):  {auc_final:.10f}")
        print(f"  grad_norm[0]:                {gn[0]:.6f}")
        print(f"  grad_norm[t*-1, t*, t*+1]:   "
              f"{gn[max(0, os_idx-1)]:.6f}, {gn[os_idx]:.6f}, "
              f"{gn[min(len(gn)-1, os_idx+1)]:.6f}")
        print(f"  grad_norm[-1]:               {gn[-1]:.6f}")
        print(f"  noise_scales: sigma1={r.history.noise_scales.sigma1:.6e}, "
              f"sigma2={r.history.noise_scales.sigma2:.6e}, "
              f"sigma2_hat={r.history.noise_scales.sigma2_hat:.6e}")
        print(f"  params_random first 3 vals:  {pytree_first_leaf_head(r.params_random)}")
        print(f"  params_final  first 3 vals:  {pytree_first_leaf_head(r.params_final)}")
        print()

    print("-" * 72)
    print("Cross-run comparison (q=100 vs q=200)")
    print("-" * 72)
    r100 = results[100]
    r200 = results[200]

    same_output_step = (r100.history.output_step == r200.history.output_step)
    print(f"  output_step matches:        {same_output_step}  "
          f"(q=100: {r100.history.output_step}, q=200: {r200.history.output_step})")

    max_diff_random = pytree_max_abs_diff(r100.params_random, r200.params_random)
    max_diff_final = pytree_max_abs_diff(r100.params_final, r200.params_final)
    print(f"  max|Δ| params_random:       {max_diff_random:.6e}")
    print(f"  max|Δ| params_final:        {max_diff_final:.6e}")

    auc_rand_100 = evaluate_auc(r100.params_random, split.x_test, split.y_test)
    auc_rand_200 = evaluate_auc(r200.params_random, split.x_test, split.y_test)
    auc_final_100 = evaluate_auc(r100.params_final, split.x_test, split.y_test)
    auc_final_200 = evaluate_auc(r200.params_final, split.x_test, split.y_test)
    auc_rand_diff = abs(auc_rand_100 - auc_rand_200)
    auc_final_diff = abs(auc_final_100 - auc_final_200)
    print(f"  |Δ AUC(w_bar)|:             {auc_rand_diff:.10f}")
    print(f"  |Δ AUC(w_T)|:               {auc_final_diff:.10f}")
    print()

    print("-" * 72)
    print("Interpretation guide")
    print("-" * 72)
    if max_diff_random == 0.0 and same_output_step:
        print("SCENARIO 3 (suspicious): identical params_random AND same "
              "output_step.")
        print("  This suggests q is not affecting the trajectory before t*,")
        print("  or the snapshot mechanism is broken. Recommend stopping and")
        print("  inspecting train.py's loop.")
    elif max_diff_random == 0.0 and not same_output_step:
        print("SCENARIO 3b (very suspicious): identical params_random despite "
              "different output_step.")
        print("  Two different snapshot indices produced identical params.")
        print("  Inspect the output-step snapshot logic in train.py.")
    elif auc_rand_diff < 1e-8 and max_diff_random > 0.0:
        print("SCENARIO 4 (very suspicious): different params_random produce "
              "identical AUC to >8 decimals.")
        print("  Essentially impossible without a bug somewhere downstream.")
    elif auc_rand_diff < 1e-4 and max_diff_random > 0.0:
        print("SCENARIO 1 or 2 (benign): trajectories differ; AUC happens to "
              "round the same at 4 decimals.")
        print("  Higher-precision AUC differs by "
              f"{auc_rand_diff:.10f}, which explains the 4-decimal collision.")
    else:
        print("UNEXPECTED: review the numbers above manually.")
        print("  The scenarios above did not match cleanly.")


if __name__ == "__main__":
    main()
