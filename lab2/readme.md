# CS3807 Deep Learning Lab — Experiment 2
### MLP for Multi-Class Image Classification (Fashion-MNIST)

Shiv Nadar University Chennai · B.Tech AI & Data Science · Semester V

## Overview

This experiment implements a Multi-Layer Perceptron (MLP) in TensorFlow/Keras to classify Fashion-MNIST images into 10 clothing categories, then uses `RandomizedSearchCV` (via SciKeras) to automatically tune hyperparameters and compares the tuned model against a manually-configured baseline.
|

## Pipeline

1. **Data** — Fashion-MNIST: 60,000 train / 10,000 test, 10 classes, 28x28 grayscale.
2. **Preprocessing** — flatten images to 784-d vectors, normalize pixels to [0, 1], one-hot encode labels.
3. **Baseline model** — `784 -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(10, Softmax)`, Adam, categorical cross-entropy, 40 epochs, batch size 32.
4. **Evaluation** — accuracy, precision, recall, F1 (macro), confusion matrix, classification report.
5. **Hyperparameter search** — `RandomizedSearchCV`, 20 sampled combinations x 5-fold CV, over:
   - hidden layers: 1-3, hidden neurons: 32/64/128/256
   - learning rate: 0.1/0.01/0.001, batch size: 16/32/64/128, epochs: 10/20/30
   - optimizer: SGD/Adam/RMSProp, activation: ReLU/Tanh/Sigmoid, dropout: 0.0/0.2/0.5
6. **Optimized model** — retrained on best-found config and evaluated on the test set.

## Key Results

| Metric | Baseline (2 hidden layers, 40 epochs) | Optimized (1 hidden layer, 20 epochs) |
|---|---|---|
| Test Accuracy | 0.8880 | 0.8834 |
| Precision (macro) | 0.8874 | 0.8881 |
| Recall (macro) | 0.8880 | 0.8834 |
| F1-score (macro) | 0.8872 | 0.8845 |
| Training time | ~216 s | 82.15 s |

**Best hyperparameters found:** 1 hidden layer, 128 neurons, ReLU, no dropout, Adam, lr = 0.001, batch size 64, 20 epochs — CV accuracy 0.8903.

**Takeaway:** the tuned model doesn't beat the baseline on raw accuracy (they're within ~0.5 points of each other), but it matches the baseline's performance with half the hidden layers and roughly a quarter of the training time. The baseline overfits after ~epoch 7 (validation loss rises while training loss keeps falling); the optimized model's shallower architecture and shorter training run make it a more efficient, less overfit choice overall. Most misclassifications across both models involve visually similar upper-body classes — Shirt, T-shirt/top, Pullover, and Coat.

## How to Run

```bash
pip install scikeras==0.13.0 scikit-learn==1.5.0
jupyter notebook DLLab2.ipynb
```

Run all cells top to bottom. The hyperparameter search cell (`RandomizedSearchCV`, 100 total fits) is the slowest step — expect it to take the longest of the notebook.

## Report

`DLLab2_report.pdf` contains the full write-up: objective, theory, dataset, preprocessing, experimental procedure, hyperparameter search discussion, classification report, all 9 mandatory plots (with placeholders for the saved PNGs), results tables, discussion Q&A, and conclusion.
