# Neural Network Models for Group Testing Data

[![Award](https://img.shields.io/badge/Award-ENAR_2026_Distinguished_Student_Paper-gold.svg)]()
[![R: 4.4.0+](https://img.shields.io/badge/R-4.4.0+-blue.svg)]()
[![C++: GCC 12.3.0+](https://img.shields.io/badge/C++-GCC_12.3.0+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

This repository contains the official implementation of the paper **"Neural Network Models for Group Testing Data"**.

## Overview
Group testing is a highly cost-effective strategy for infectious disease screening. However, modeling multiplex group testing data is mathematically challenging due to the latent nature of individual statuses, the complexity of pooled testing decoding (e.g., Dorfman algorithms), and potential class imbalance in low-prevalence settings.

This repository provides a novel, high-performance computational framework using **Neural Networks combined with an Expectation-Maximization (EM) algorithm**. To maximize computational efficiency, the neural network architecture, forward/backward propagation, and log-likelihood optimizations are implemented entirely from scratch in **C++** via `Rcpp` and `RcppArmadillo`, seamlessly integrated into an R-based EM framework.

### Key Features
- **Latent Status Modeling:** Neural network architectures designed to approximate the individual infection probabilities directly from pooled, group-level testing data.
- **Monte Carlo EM Framework:** Robust latent variable sampling (`SampLatent`) and EM-based training strategies to handle unobserved true individual statuses.
- **Imbalance Handling:** Built-in dynamic sample re-weighting mechanisms (Weighted Cross-Entropy Loss) tailored for highly imbalanced datasets where disease prevalence is extremely low.
- **High-Performance Computing:** Core bottleneck computations are unified in highly optimized C++ to handle large-scale simulation studies seamlessly.

## Repository Structure
```text
Neural-Network-Models-for-Group-Testing-Data/
├── src/                  
│   └── GroupTesting.cpp  # Unified C++ core (Log-likelihood, NN with dynamic weights, Latent sampling)
├── utils.R               # Utility functions (Dorfman decoding, data generation, etc.)
├── main.R                # Main entry script for the EM algorithm and model training
└── run_simulation.sh     # SLURM submission script for HPC Array Jobs
```

## Simulation Scenarios
The `main.R` script is designed to be highly modular, supporting various underlying data generation mechanisms, prevalence distributions, and fitting algorithms:

**Data Generation Models (`data_model`):**
* *Standard Feature Simulations:*
  * `M1`: Linear additive model (main effects only).
  * `M2`: Model with high-order interaction terms.
  * `M3`: Highly non-linear model featuring sine functions and interactions.
* *Prevalence & Imbalance Simulations:*
  * `B`: Balanced dataset scenario.
  * `IM1`: Imbalanced dataset scenario (low disease prevalence).
  * `IM2`: Highly imbalanced dataset scenario (extremely low disease prevalence).

**Algorithms (`algorithm`):**
* *Traditional Baselines:*
  * `L1`: Standard Logistic Regression (main features).
  * `L2`: Full-interaction Logistic Regression (15 features including two-way to four-way interactions).
* *Neural Network Models:*
  * `NN` / `CEL`: Multi-layer Feedforward Neural Network using standard Cross-Entropy Loss (Current setup: 4 layers, [50, 50, 50, 1] nodes).
  * `WCEL`: Neural Network trained with **Weighted Cross-Entropy Loss**, applying data-driven density weights to dynamically penalize misclassification of the minority class in low-prevalence settings.

## Usage
Ensure you have **R (>= 4.4.0)** and a **C++ Compiler** (e.g., GCC >= 12.3.0) installed. The required R packages include `Rcpp` and `RcppArmadillo`. Note that `src/GroupTesting.cpp` will be automatically compiled via `sourceCpp()` during the first run.

### Running Locally
You can run a single task via the command line by specifying the data model, algorithm, and task ID:
```bash
# Example 1: Standard simulation with Neural Network
Rscript main.R M2 NN 1

# Example 2: Highly imbalanced data simulation with WCEL
Rscript main.R IM2 WCEL 1
```

### Running on HPC Clusters (SLURM)
For large-scale simulations, we provide a Bash script configured for SLURM workload managers. It utilizes Array Jobs with built-in jittering to prevent I/O storms and memory crashes.
```bash
sbatch run_simulation.sh
```

## Upcoming Modules
This repository is actively being updated. The following module will be released shortly:
- [ ] **Iowa Chlamydia Dataset Analysis:** Real-world application of the proposed framework on the Iowa State Chlamydia screening data.
