# MolP-PC

MolP-PC is a deep learning framework for multi-view feature fusion and multi-task learning. The model comprehensively captures molecular features by integrating 1D molecular fingerprints, 2D molecular graphs, and 3D geometric representations, utilizing an attention-gated mechanism for multi-view feature fusion, and combining it with a self-adaptive multi-task learning strategy to enhance the predictive performance of tasks with limited samples. The model structure includes modules for **multi-view feature extraction**, **multi-view feature fusion**, and **multi-task prediction**, supporting efficient molecular property prediction and suitable for scenarios with sparse data.

# Usage

## 1.Installation

`python=3.8`

`dgl=0.9.1post1`

`pytorch=1.10.1`

`rdkit`

## 2.Data preparation

Before training, you need to build a training dataset using following code:

`python build_dataset.py /path/to/data file1.csv file2.csv file3.csv`

## 3.Experiment

The `experiment.py` script is designed to perform model training and inference.