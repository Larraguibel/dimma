# dimma

JAX-based library of differentially private optimization algorithms.

## Installation

Base install (library only):

```bash
pip install -e .
```

To also run the test suite, install the `dev` extras:

```bash
pip install -e ".[dev]"
```

To run the notebooks under [examples/](examples/) (e.g. the Criteo
experiments), install the `examples` extras:

```bash
pip install -e ".[examples]"
```

You can combine extras, e.g. `pip install -e ".[dev,examples]"`.

### Dependency overview

- **Runtime** (installed by default): `jax`, `flax`, `optax`,
  `dp-accounting`, `pandas`, `pyarrow`, `numpy`.
- **`dev` extras**: `pytest`, `scikit-learn` (used by the regression
  tests against reference implementations).
- **`examples` extras**: `matplotlib`, `scikit-learn`, `jupyter`
  (needed by the notebooks and plotting utilities in
  [examples/criteo/](examples/criteo/)).

## Datasets and licensing

`dimma.datasets` provides convenience loaders for canonical benchmark
datasets. Datasets are downloaded on demand to a user-controlled cache
directory (see ``dimma.datasets._cache.get_cache_dir``).

Each dataset retains its original license:

- **Criteo 1M** (`dimma.datasets.load_criteo`): CC-BY-NC-SA 4.0. Original
  data © Criteo Labs. Non-commercial use only. Derivative works must
  be shared under the same license.

The library itself does not change these licenses. Users of `dimma` are
responsible for complying with the license of each dataset they load.
The library prints a one-time attribution notice to stderr on first
download per process.

## JAX Version

JAX version is **not pinned** by this library. Manage JAX (and its CUDA variant if
applicable) in your own environment before installing dimma. See the
[JAX installation guide](https://github.com/google/jax#installation).

## Development

This project uses a `src/` layout. After cloning, install in editable mode:

```bash
pip install -e ".[dev]"
pytest tests/
```
