# Approximate Inference

This repository contains a modular and clean implementation of Gaussian Process Regression, Mean-Field Variational Inference, and Loopy Belief Propagation.

## Project Structure

```
core/
├── data/                      # Data loading and synthetic data generation
├── models/                    # Core model implementations
│   ├── gaussian_process/      # Gaussian Process Regression
│   ├── mean_field/            # Mean-Field Variational Inference
│   └── loopy_bp/              # Loopy Belief Propagation
└── scripts/                   # Example training and evaluation scripts
```

## Features

- **Gaussian Process Regression**: Modular GP implementation with pluggable kernels and numerical stability.
- **Mean-Field Variational Inference**: Fully factorized approximation on binary factor graphs.
- **Loopy Belief Propagation**: Parallel message passing for approximate marginal inference.

## ▶️ How to Run

Example scripts are under `core/scripts/`. They’re minimal and to the point.

```bash
# Run GP regression
python core/scripts/gp_regression_test.py

# Run Mean-Field
python core/scripts/mean_field_approximation.py

# Run Loopy Belief Propagation
python core/scripts/loopy_bp_test.py
```


## Notes

- Kernels and model structures are kept modular for reuse and experimentation.
- Numerical precision and stability are prioritized for inference tasks.

