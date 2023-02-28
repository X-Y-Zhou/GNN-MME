# GNN-MME

[![DOI](https://zenodo.org/badge/605591822.svg)](https://zenodo.org/badge/latestdoi/605591822)

---

This repository contains the julia code and parameters corresponding to the effeciency test presented in:
Efficient and scalable prediction of spatio-temporal stochastic gene expression in cells and tissues using graph neural networks

**File descriptions:**

- "Fig2a-c/data/2_cells_v1/proba/cell_1.csv" is the training data
- "Fig2a-c/data/2_cells_v1/proba/cell_1_pred.csv" is the predicted distribution
- "Fig2a-c/Birth_Death_train.jl" is the training code.
- "Fig2a-c/Birth_Death_valid.jl" is the predicting code for cell "E","C","U","S" and "T".
- "p.bson" stores the trained parameters
- "Results" summarizes the predicted results for Fig.2, FigS4 and FigS5 (inset).

**Requirements:**

- Julia >= 1.6.5
- Flux v0.12.8
- DifferentialEquations v7.0.0
- DiffEqSensitivity v6.66.0
- Zygote v0.6.33

**The method is well described in:**

* Z. Cao _et. al._ [Efficient and scalable prediction of spatio-temporal stochastic gene expression in cells and tissues using graph neural networks]().
