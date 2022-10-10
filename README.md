# Learning about GNN

This repository contains some tutorials and experiments to learn about Graph Neural Networks, using PyTorch Geometric as
the main framework. Currently, only one learning task is covered:

 - graph classification

# Project structure
Each folder under `experiments` contains a `main.py` file which runs the corresponding experiment. The
`main` file imports classes from:
 - `src/<experiment>/utils` for generic data loading & preprocessing
 - `src/<experiment>/datasets` for functions specific to a certain dataset
 - `src/<experiment>/models` for model initialization, training & inference
