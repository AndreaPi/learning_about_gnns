# Learning about GNN


[![dependencies](https://img.shields.io/badge/dependencies-apache%20or%20better-01A5B8)](https://github.build.ge.com/AILAB/ailab/blob/main/open-source-license.png)

This repository contains some tutorials and experiments to learn about Graph Neural Networks, using PyTorch Geometric as
the main framework. Currently, only one learning task is covered:

 - graph classification


# Project structure
Each folder under `experiments` contains a `main.py` file which runs the corresponding experiment. The
`main` file imports classes from:
 - `src/<experiment>/utils` for generic data loading & preprocessing
 - `src/<experiment>/datasets` for functions specific to a certain dataset
 - `src/<experiment>/models` for model initialization, training & inference
