# CS3807 - Deep Learning Lab - Experiment 5

Study of CNN training using MobileNetV2 on the Oxford-IIIT Pet
dataset. Covers weight initialization, regularization, batch
normalization, optimizers, hyperparameter tuning, transfer learning,
fine-tuning, and model selection using 5-fold cross-validation.

## What's in here

- Weight initialization comparison (Zero, Random, Xavier, He)
- Regularization comparison (None, L2, Dropout, BatchNorm)
- Optimizer comparison (SGD, Momentum, RMSProp, Adam)
- Hyperparameter sweep (learning rate, batch size, dropout rate)
- Transfer learning with MobileNetV2 (feature extraction + fine-tuning)
- 5-fold cross-validation for final model selection
- Final evaluation on a held-out test set (confusion matrix, precision/recall/F1)

## Dataset

Oxford-IIIT Pet Dataset - 37 cat and dog breeds. Images are resized to
224x224x3 and normalized to match what the pretrained MobileNetV2
expects. The dataset can be loaded through `tensorflow-datasets`
(`oxford_iiit_pet`) or downloaded directly from the
[official page](https://www.robots.ox.ac.uk/~vgg/data/pets/).

## Setup

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project structure

```
Ex5/
├── README.md
├── requirements.txt
├── experiment_5_student.tex     # lab report (Overleaf/LaTeX)
├── figures/                     # all plots referenced in the report
│   ├── plot1_init_loss.png
│   ├── plot2_init_valacc.png
│   ├── ...
│   └── plot14_confusion_matrix.png
└── src/                          # training/evaluation scripts (if included)
```

## How to run

1. Install dependencies from `requirements.txt`.
2. Load and preprocess the Oxford-IIIT Pet dataset (resize to
   224x224, normalize for MobileNetV2).
3. Run each experiment stage in order:
   - initialization comparison
   - regularization comparison
   - optimizer comparison
   - hyperparameter sweep
   - transfer learning (feature extraction, then fine-tuning)
   - 5-fold cross-validation on the shortlisted configurations
   - final retrain + one-time test set evaluation
4. Plots get saved to `figures/` and match the plot numbers used in
   the report.

> Note: keep the test set completely separate until the very last
> step. It should never be touched while comparing configurations or
> during cross-validation.

## Report

The full write-up with all plots, tables, and discussion answers is
in `experiment_5_student.tex`, ready to compile in Overleaf (just make
sure the `figures/` folder sits alongside the `.tex` file).

## Requirements

See `requirements.txt`. Main dependencies: TensorFlow, scikit-learn,
matplotlib, seaborn, pandas, Pillow, tensorflow-datasets.
