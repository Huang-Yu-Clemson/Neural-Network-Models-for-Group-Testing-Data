# Neural Network Models for Group Testing Data

[![Award](https://img.shields.io/badge/Award-ENAR_2026_Distinguished_Student_Paper-gold.svg)]()
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

This repository contains the official PyTorch implementation of the paper **"Neural Network Models for Group Testing Data"**.

## 📌 Overview
Group testing is a cost-effective strategy for infectious disease screening. However, modeling multiplex group testing data is challenging due to the latent individual statuses and potential class imbalance. 

This repository provides a novel framework using neural networks combined with an **Expectation-Maximization (EM) algorithm** and **data augmentation** strategies to efficiently train models directly on group-level testing data. 

### Key Features:
- **Latent Status Modeling:** Neural network architectures designed to approximate the joint probabilities of multiplex individual statuses.
- **EM Framework:** Robust data augmentation and EM-based training strategies.
- **Imbalance Handling:** Built-in re-weighting mechanisms to handle low prevalence rates.

## 📂 Repository Structure
```text
├── data/                   # Sample datasets and data generation scripts
├── models/                 # Neural network architectures (e.g., Softmax NN)
├── utils/                  # Helper functions for EM algorithm and data augmentation
├── main.py                 # Main script to train and evaluate the model
├── requirements.txt        # List of dependencies
└── README.md
