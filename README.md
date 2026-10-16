# Budbæreren

*Budbæreren* (Norwegian: "the messenger") is a small neural-network library
written from scratch in pure Python + NumPy, and a classifier built on it that
**prunes its own connections during training** under a hard sparsity budget.

> Status: under construction. See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)
> for the phase-by-phase build log.

## Library policy

- **NumPy** is used throughout; **Matplotlib** for plots.
- **scikit-learn is used only to download the MNIST dataset**
  (`sklearn.datasets.fetch_openml`). It is not used for training, models,
  preprocessing, metrics or gradients.
- No PyTorch, TensorFlow, JAX, Keras, HuggingFace, SciPy, or any autodiff /
  pruning / neural-network library. The autodiff engine, backward pass,
  pruning mechanism, optimizers and training loop are all in `budbaereren/`.

## Install

```bash
uv sync
uv run pytest
```
