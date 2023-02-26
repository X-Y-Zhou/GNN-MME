# GEM-GNN
[![DOI](https://zenodo.org/badge/605591822.svg)](https://zenodo.org/badge/latestdoi/605591822)

***
This repository contains the julia code and parameters corresponding to the effeciency test presented in:
Efficient and scalable prediction of spatio-temporal stochastic gene expression in cells and tissues using graph neural networks

File descriptions:

- "Figure2_a-c/data/2_cells_v1/proba/cell_1.csv" is the source data
- "Figure2_a-c/data/2_cells_v1/proba/cell_1_pred.csv" is the predicted data
- "Figure2_a-c/Birth_Death_train.jl" is the code for training.
- "Figure2_a-c/Birth_Death_valid.jl" is the code for predicting Cell "ECUST".
- "p.bson" is the trained parameters
- "results" summarizes the experimental data from literature for Fig. 2,FigS3 (inset)

Requirements:

- Julia >= 1.4.2
- Flux v0.13.5
- DifferentialEquations v7.2.0
- DiffEqSensitivity v6.79.0
- Zygote v0.6.46

**The method is well described in:**
* Z. Cao _et. al._ [Efficient and scalable prediction of spatio-temporal stochastic gene expression in cells and tissues using graph neural networks]().
   
