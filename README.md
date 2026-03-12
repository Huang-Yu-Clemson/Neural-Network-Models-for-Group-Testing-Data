# Neural Network Models for Group Testing Data

[![Award](https://img.shields.io/badge/Award-ENAR_2026_Distinguished_Student_Paper-gold.svg)]()
[![R: 4.4.0+](https://img.shields.io/badge/R-4.4.0+-blue.svg)]()
[![C++: GCC 12.3.0+](https://img.shields.io/badge/C++-GCC_12.3.0+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

This repository contains the official implementation of the paper **"Neural Network Models for Group Testing Data"**.

## Overview
Group testing is a highly cost-effective strategy for infectious disease screening. However, modeling multiplex group testing data is mathematically challenging due to the latent nature of individual statuses, the complexity of pooled testing decoding (e.g., Dorfman algorithms), and potential class imbalance in low-prevalence settings.

This repository provides a novel, high-performance computational framework using **Neural Networks combined with an Expectation-Maximization (EM) algorithm**. To maximize computational efficiency, the neural network architecture, forward/backward propagation, and log-likelihood optimizations are implemented entirely from scratch in **C++** via `Rcpp` and `RcppArmadillo`, seamlessly integrated into an R-based EM framework.

### Key Features:
- **Latent Status Modeling:** Neural network architectures designed to approximate the individual infection probabilities directly from pooled, group-level testing data.
- **Monte Carlo EM Framework:** Robust latent variable sampling (`SampLatent`) and EM-based training strategies to handle unobserved true individual statuses.
- **High-Performance Computing:** Core bottleneck computations (e.g., log-likelihood estimation, NN gradient updates) are written in optimized C++ to handle large-scale simulation studies.
- **Imbalance Handling (Upcoming):** Built-in re-sampling and weighting mechanisms tailored for highly imbalanced datasets with low disease prevalence.

## Repository Structure
```text
Neural-Network-Models-for-Group-Testing-Data/
├── src/                  # Core C++ modules for speed acceleration
│   ├── loglik.cpp        # Log-likelihood computation for pooled data
│   ├── nn.cpp            # Custom Neural Network implementation (RcppArmadillo)
│   └── SampLatent.cpp    # Latent variable sampling for the E-step
├── utils.R               # Utility functions (Dorfman decoding, data generation, etc.)
├── main.R                # Main entry script for the EM algorithm and model training
└── run_simulation.sh     # SLURM submission script for HPC Array Jobs

## Simulation Scenarios
The `main.R` script is designed to be highly modular, supporting various underlying data generation mechanisms and fitting algorithms:

**Data Models (`data_model`):**
* `M1`: Linear additive model (main effects only).
* `M2`: Model with high-order interaction terms.
* `M3`: Highly non-linear model featuring sine functions and interactions.

**Algorithms (`algorithm`):**
* `L1`: Standard Logistic Regression (4 main features).
* `L2`: Full-interaction Logistic Regression (15 features including two-way to four-way interactions).
* `NN`: Multi-layer Feedforward Neural Network (Current setup: 4 layers, `[50, 50, 50, 1]` nodes, ReLU & Sigmoid activations).

## Usage
Ensure you have **R (>= 4.4.0)** and a **C++ Compiler** (e.g., GCC >= 12.3.0) installed. The required R packages include `Rcpp` and `RcppArmadillo`. Note that `src/*.cpp` files will be automatically compiled via `sourceCpp()` during the first run.

### Running Locally
You can run a single task via the command line by specifying the data model, algorithm, and task ID:
```bash
Rscript main.R M2 NN 1
