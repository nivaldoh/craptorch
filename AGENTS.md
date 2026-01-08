# Repository Guidelines

This repository implements a minimal PyTorch-like core in pure Python with NumPy, plus a focused pytest suite. Follow the guidance below to keep contributions consistent and easy to review.

## Project Structure & Module Organization
- `craptorch/`: package root.
- `craptorch/core/`: core building blocks:
  - `tensor.py` (Tensor type and math)
  - `autograd.py` (autograd primitives)
  - `layers.py`, `activations.py`, `losses.py` (NN components)
- `tests/`: pytest suite grouped by domain (`tests/tensor`, `tests/activations`, `tests/layers`, `tests/losses`, `tests/autograd`).
- `scripts/`: useful entrypoint scripts that may have standalone functionality.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install NumPy + pytest tooling.
- `pytest`: run the full test suite.
- `pytest tests/tensor -k matmul`: run a focused subset by path/keyword.
- `pytest --cov=craptorch --cov-report=term-missing`: optional coverage report.
- Environment: README references a shared venv; activate it with `source ../tinytorch/.venv/bin/activate` or use your own local venv.

## Coding Style & Naming Conventions
- Indentation: 4 spaces; follow existing formatting in `craptorch/core`.
- Names: classes in `CamelCase` (e.g., `Tensor`), functions/variables in `snake_case`, constants in `UPPER_SNAKE_CASE`.
- Keep APIs NumPy-friendly (use `np.ndarray` internally) and avoid extra dependencies.

## Testing Guidelines
- Framework: pytest (with `pytest-cov` available).
- File naming: `test_*.py` and frequently `test_unit_*` for unit tests.
- Prefer colocating tests by domain (e.g., tensor ops in `tests/tensor/`).
- Add/extend tests for new ops, edge cases, and error paths.

## Commit & Pull Request Guidelines
- Commit messages are short, lowercase, and descriptive (e.g., `activation tests`, `matmul test`, `mse`); follow that pattern.
- PRs should include a concise summary, tests run, and any API changes or behavior notes. Link related issues if applicable.
