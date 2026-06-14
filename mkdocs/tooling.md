# JAX, Flax and Optax Primitives

!!! note

    This page covers the JAX / Flax / Optax tooling we use to build non-standard DP-SGD variants. It assumes you already understand what per-sample gradients, clipping, and the Gaussian mechanism are. If not, learn about that, and check [Differential Privacy for SGD: Overview](overview.md) first.

## Why JAX for this project

DP-SGD variants need direct control over primitives that high-level libraries hide or fix. OpenDP and similar libraries, at least up to the date (May 2026)  focus on tabular data DP, not SGD. The ones that do focus on DP-SGD, such as Opacus, Keras-DP, and `optax.contrib.dpsgd` all assume the same rigid pipeline: batch sapling (poisson or shuffling) → per-sample grads → clip to a single norm `C` → average → add Gaussian noise → optimizer update. Anything outside that shape is awkward or impossible to express.

Our algorithms break that shape. For example, **Private SpiderBoost** alternates between two kinds of steps in a single training loop: *anchor* steps that compute a noisy gradient on a large batch, and *variation* steps that update a variance-reduced gradient estimate using the *difference* of gradients across consecutive iterates, with noise scaled to that difference rather than to a fixed sensitivity. Two update rules, two noise scales, one training loop. See [Private Spider-Boost](algorithms/spiderboost/index.md) for details.

Another example is algorithms that require truncated Poisson subsampling. In standard Poisson subsampling, each example is included in a batch independently with probability *p,* making the batch size random. Truncated Poisson adds a cap: if the drawn batch exceeds a maximum size, it is subsampled down to that cap. Applying this cap means intervening inside the batch construction step. A premade batch-handling function does not expose that step, so the only options are to wrap it (paying redundant computation) or to reimplement it. This is why the sampling procedure needs to be written into our own training loop rather than delegated to a library.

No single Optax optimizer can express this, and no Opacus hook lets you in deep enough to inject it cleanly. JAX is the right tool because it exposes the primitives directly: per-sample gradients, custom batch handling, explicit randomness, and a training loop we write ourselves.

## JAX: the four primitives you actually need

If you understand `grad`, `vmap`, `jit`, and `random`, you have enough JAX to implement any DP-SGD variant. The rest is library convenience.

### `jax.grad` — scalar-output autodiff

Differentiates a function returning a scalar with respect to its first argument. Unlike PyTorch's `.backward()`, which mutates `.grad` attributes as a side effect, `jax.grad` returns the gradient as a value. Code becomes a chain of pure functions, not a graph of mutable tensors.

```python
def loss(params, x, y):
    return jnp.mean((model_apply(params, x) - y) ** 2)

grad_fn = jax.grad(loss)         # function: (params, x, y) -> grad
g = grad_fn(params, x_batch, y_batch)  # mean gradient over the batch
```

Note this gives you the **mean (or sum) gradient over the batch**, not per-sample gradients. That's what `vmap` is for.

### `jax.vmap` — the per-sample gradient trick

This is *the* reason JAX is the natural fit for DP-SGD. `vmap` vectorises a function over an extra axis without writing a Python loop and without retracing.

```python
def per_sample_loss(params, x, y):
    # x, y are SINGLE examples, no batch dimension
    return loss_on_one_example(params, x, y)

per_sample_grad = jax.vmap(jax.grad(per_sample_loss), in_axes=(None, 0, 0))
# in_axes=(None, 0, 0): params shared, x and y batched on axis 0

g_per_sample = per_sample_grad(params, x_batch, y_batch)
# g_per_sample is a pytree with leading batch dim B on every leaf
```

The composition `vmap(grad(...))` is the canonical DP-SGD pattern. Each sample's gradient is computed independently, and you get back a pytree of shape `(B, *param_shape)` per parameter. From there you can clip per-sample, sum, and add noise.

Performance note: under XLA this is fused and roughly as fast as a single batched forward/backward in most cases. It is dramatically faster than a Python loop over samples and avoids the "expanded weights" / hooks machinery Opacus uses in PyTorch.

### `jax.jit` — compile to XLA

Wraps a function so its first call traces and compiles it; subsequent calls with the *same input shapes and dtypes* hit the compiled version. Standard speedup, standard story.

The gotcha that matters for DP: **tracing is shape-specialised**. If your batch size changes across iterations, `jit` retraces every time, which kills performance. This is exactly the situation under Poisson subsampling, where the realised batch size is `Binomial(n, p)`.

The standard fix is to pad the batch to a fixed maximum size and mask the unused slots, then account for the mask when summing per-sample gradients. The maximum-size design has implications for the privacy accounting (effective sampling rate, group privacy if the bound is occasionally exceeded). Details are deferred to the implementation-specific pages.

### `jax.random` — explicit PRNG keys

JAX has no global random state. Every random operation takes an explicit `key`, and keys must be split before reuse:

```python
key = jax.random.key(0)
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape=(d,))
```

For DP this is a feature, not a chore. The noise added to a gradient is the privacy mechanism. Having the key as an explicit argument means:

- noise generation is reproducible exactly (auditable runs),
- it is impossible to accidentally reuse the same noise across steps (which would break the DP guarantee),
- you can isolate the noise key from the sampling key cleanly.

Pattern we use throughout: at every training step, split the loop key into `(noise_key, sampling_key, next_key)` and return `next_key` as part of the carried state.

## Flax NNX: pytrees, classes, and the split / merge pattern

JAX functions are pure: they take pytrees of arrays in and return pytrees of arrays out. A neural network's parameters form a pytree naturally. But we also want a model to feel like an object with methods, layers, and mutable state, the way PyTorch lets you write it. Reconciling those two views is what Flax NNX is for.

### The two views: object vs. pytree

Flax NNX gives you a `Module` class that you write the normal Python way:

```python
import jax, jax.numpy as jnp
from flax import nnx

class MLP(nnx.Module):
    def __init__(self, din, dhidden, dout, *, rngs: nnx.Rngs):
        self.lin1 = nnx.Linear(din, dhidden, rngs=rngs)
        self.lin2 = nnx.Linear(dhidden, dout, rngs=rngs)

    def __call__(self, x):
        return self.lin2(jax.nn.relu(self.lin1(x)))

model = MLP(10, 32, 1, rngs=nnx.Rngs(0))
y = model(jnp.ones((4, 10)))   # call it like a normal Python object
```

Internally, NNX stores parameters as `nnx.Param` variables attached to the module. So far this looks like PyTorch. The trick is that NNX can convert this object on demand into a pure pytree:

```python
graphdef, state = nnx.split(model)
# graphdef: the static structure (layer types, shapes, connectivity)
# state:    a pure pytree of Param values, ready for JAX transforms

model_again = nnx.merge(graphdef, state)  # reconstruct the object
```

`split` decomposes the module into a *static* part (`graphdef`, treated as a hashable constant by JAX) and a *dynamic* part (`state`, a pytree of arrays JAX is happy to transform). `merge` is the inverse.

### Why this matters for DP-SGD

`jax.vmap` and `jax.grad` require pure functions over pytrees. Per-sample gradients via `vmap(grad(loss_fn))` need `loss_fn` to be a pure function of `(params, x, y)`. NNX's split/merge gives you exactly that without forcing you to write the model as a flat function.

The pattern looks like:

```python
graphdef, params = nnx.split(model)

def per_sample_loss(params, x, y):
    model = nnx.merge(graphdef, params)   # graphdef closed over as constant
    pred = model(x)
    return loss_on_one_example(pred, y)

per_sample_grad_fn = jax.vmap(jax.grad(per_sample_loss), in_axes=(None, 0, 0))
g_per_sample = per_sample_grad_fn(params, x_batch, y_batch)
```

After the parameter update, write the new params back with `nnx.update(model, new_params)` and the object-style `model(x)` keeps working.

!!! note

    **Implementation knowledge, current as of writing:** the API above (`nnx.split`, `nnx.merge`, `nnx.update`, `nnx.Param`, `nnx.Rngs`) is from current Flax NNX docs (`flax.readthedocs.io/en/latest`). NNX is recent and evolving. If you find a tutorial using `model.split()` as a method, or a different `State` shape, check the version. When library behaviour matters for correctness, read the source rather than trusting older posts.

### Filtered splits

`nnx.split` can split state by variable type, which is useful when you want to separate trainable parameters from buffers, batch-norm statistics, or anything else you want to handle differently:

```python
graphdef, params, batch_stats = nnx.split(model, nnx.Param, nnx.BatchStat)
```

Only `params` should pass through the DP-SGD pipeline (per-sample grad → clip → noise). Batch statistics and other non-private state are handled separately. This is the place where you decide what is and isn't privatised; get it wrong here and the privacy guarantee is wrong.

## Optax: what we use and what we deliberately don't

Optax is the gradient transformation library. Mental model: an Optax optimizer takes a *single* gradient pytree and returns an update.

**Use Optax for the post-aggregation update.** Once you have a single noised, aggregated gradient, Optax handles the rest cleanly: SGD with momentum, Adam, learning rate schedules, weight decay, gradient accumulation across micro-batches, etc.

```python
import optax
tx = optax.adam(1e-3)
opt_state = tx.init(params)

# ... compute private_grad via vmap + clip + sum + noise ...

updates, opt_state = tx.update(private_grad, opt_state, params)
new_params = optax.apply_updates(params, updates)
```

**Do not use Optax on the per-sample side.** Anything that touches per-sample gradients (clipping, noise calibration, variance-reduced gradient estimators like SpiderBoost's variation step) is written by hand. Optax's gradient transforms assume a single aggregated gradient and cannot see per-sample structure.

`optax.contrib.dpsgd` exists. It implements the standard pipeline, just like Opacus does for PyTorch. We do not build on it for the same reason we do not build on Opacus: it forecloses exactly the variations we want to study. Apply the heuristic above — it assumes the aggregated-gradient shape — and move on.

## A minimal end-to-end DP-SGD step

This is the skeleton every algorithm in the project specialises. It implements **standard DP-SGD (Abadi et al., 2016)** with the following assumptions:

- **Fixed batch size** `B` drawn by uniform shuffling (not Poisson subsampling).
- **Single clipping norm** `C` applied uniformly to all per-sample gradients.
- **Gaussian mechanism** on the *sum* of clipped gradients, noise std = `sigma * C`.
- **SGD** as the post-aggregation optimizer (swap in Adam or anything else via Optax).

None of these are required by the codebase. They are the baseline. Every variant deviates from one or more of them.

`MLP` is the class defined in the Flax NNX section above. In a real project file it would be imported from your model module; it is defined inline here for readability.

```python
import jax, jax.numpy as jnp
from flax import nnx  # MLP defined as nnx.Module subclass above
import optax

# --- setup (once, outside the training loop) ---
model = MLP(din=10, dhidden=32, dout=1, rngs=nnx.Rngs(0))
graphdef, params = nnx.split(model)
tx = optax.sgd(learning_rate=0.01)
opt_state = tx.init(params)

C     = 1.0   # per-sample clipping norm; sets the L2 sensitivity of the sum
sigma = 1.1   # noise multiplier; noise std on the SUM is sigma * C
key   = jax.random.key(0)

# --- per-sample loss (single example, no batch dim) ---
def per_sample_loss(params, x, y):
    model = nnx.merge(graphdef, params)  # graphdef closed over as static constant
    pred = model(x)
    return jnp.sum((pred - y) ** 2)     # scalar; sum not mean, vmap handles the batch

per_sample_grad_fn = jax.vmap(jax.grad(per_sample_loss), in_axes=(None, 0, 0))

# --- one DP-SGD step ---
@jax.jit
def dp_sgd_step(params, opt_state, key, x_batch, y_batch):
    B = x_batch.shape[0]

    # 1. per-sample gradients -- pytree with leading dim B on every leaf
    g_ps = per_sample_grad_fn(params, x_batch, y_batch)

    # 2. per-sample L2 norms
    leaves = jax.tree.leaves(g_ps)
    sq_norms = sum(jnp.sum(g.reshape(B, -1) ** 2, axis=1) for g in leaves)
    norms = jnp.sqrt(sq_norms)                        # (B,)
    scale = jnp.minimum(1.0, C / (norms + 1e-12))    # (B,)

    # 3. clip and SUM (noise is calibrated to the sum, not the mean)
    def clip_and_sum(g):
        s = scale.reshape((-1,) + (1,) * (g.ndim - 1))
        return jnp.sum(s * g, axis=0)
    g_sum = jax.tree.map(clip_and_sum, g_ps)

    # 4. Gaussian noise -- sensitivity of the sum is C by construction
    key, noise_key = jax.random.split(key)
    leaves_noise = jax.random.split(noise_key, len(jax.tree.leaves(g_sum)))
    noise_tree = jax.tree.unflatten(
        jax.tree.structure(g_sum),
        [sigma * C * jax.random.normal(k, g.shape)
         for k, g in zip(leaves_noise, jax.tree.leaves(g_sum))]
    )
    g_priv = jax.tree.map(lambda g, n: g + n, g_sum, noise_tree)

    # 5. normalise to a mean gradient and update via Optax
    g_priv = jax.tree.map(lambda g: g / B, g_priv)
    updates, opt_state = tx.update(g_priv, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, key
```

Every variant in the project modifies one or more of these five steps. SpiderBoost replaces step 1-3 on variation steps with a difference-of-gradients estimator whose sensitivity scales with consecutive iterate distance. DP-FTRL replaces step 4 with a tree aggregation mechanism. Sparse-gradient methods restrict step 4 to the gradient support. The skeleton is the contract; the experiments are deviations from it.

!!! warning

    **Conventions to pin before connecting to an accountant:**

    - Noise is on the **sum**, std = `sigma * C`. The mean-gradient std is `sigma * C / B`. Many papers write the mechanism in terms of the mean; make sure your accountant matches.
    - `B` is **fixed** here (shuffled minibatch). Under Poisson subsampling `B` is random and the noise does not change per step, but the accounting does. Different accountant required.
    - Step 4 now splits a fresh subkey per leaf. If you simplify back to one key for the whole pytree (as in a quick experiment), note that the noise on different parameter tensors becomes correlated, which is fine in practice but worth knowing.

## Pointers for going deeper

- JAX: [docs.jax.dev](http://docs.jax.dev)
- Flax NNX: [flax.readthedocs.io/en/latest/api_reference/flax.nnx](http://flax.readthedocs.io/en/latest/api_reference/flax.nnx)
- Optax: [optax.readthedocs.io](http://optax.readthedocs.io)
- DP context for this project: [Differential Privacy for SGD: Overview](overview.md)
- A concrete worked example: [Private Spider-Boost](algorithms/spiderboost/index.md)

Privacy accounting is not covered here. Accounting decisions (which accountant, sampling-amplification assumptions, composition) live with the algorithm-specific pages.

## From patterns to a library

The patterns above (per-sample `vmap(grad)`, padded Poisson masks, JIT-compiled step kernels, sum + noise on the sensitivity, post-aggregation update) are the contract every DP-SGD variant in this project follows. They are codified as reusable primitives in **dimma**, the project's JAX library. The minimal end-to-end step above is essentially what `dimma.algorithms.spiderboost.kernels` does, generalised so that the algorithm-specific math (anchor vs. variation, two noise scales, gradient-difference clipping) sits on top of the same primitives.

See [dimma: the library](library.md) for the library's module map, design conventions, and how to add a new algorithm.
