# Lab 6 — End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai
**Guru Divya Darshini U** · Reg. No. 24011101032

Full implementation of Experiment 6: sequence classification on the UCI HAR dataset with SimpleRNN, LSTM and GRU, a CNN+LSTM/GRU pipeline for video action recognition on UCF101, and an encoder–decoder LSTM for a sequence-reversal task.

## Contents

```
lab6/
├── Experiment6_RNN_LSTM_GRU.ipynb   # main notebook — run top to bottom
├── figures/                         # generated plots (created on run)
└── README.md
```

## What's inside the notebook

| Part | What it covers |
|---|---|
| Data pipeline | Raw UCI HAR inertial signals → (N, 128, 9) tensors, stratified 70/15/15 split, train-only normalization |
| Plot 1 | Sensor signal vs. time across all six activity classes |
| BPTT demo | Manual vs. Keras numerical check; vanishing/exploding gradient illustration |
| RNN / LSTM / GRU | Identical architecture and training protocol, swapping only the recurrent layer |
| Plots 2–6 | Loss/accuracy curves, confusion matrices, model comparison, sequence-length study (T = 32/64/128) |
| Video pipeline | Frozen MobileNetV2 (1280-d features) → LSTM/GRU → 5-class UCF101 action recognition, with Plots 7–9 |
| Seq2Seq | Encoder–decoder LSTM trained with teacher forcing on a synthetic sequence-reversal task |
| Extras | Unit-count sweep (16/32/64), stacked recurrent layers, BiLSTM vs LSTM, variable-length seq2seq |

## How to run

1. Open `Experiment6_RNN_LSTM_GRU.ipynb` in Google Colab.
2. `Runtime → Change runtime type → T4 GPU`.
3. `Runtime → Run all`.

The notebook downloads the UCI HAR dataset and a UCF101 video subset automatically — no manual data setup needed.

### Configuration

Key parameters are set near the top of the notebook:

```python
N_TOTAL   = 3000   # HAR windows sampled (None = full dataset)
UNITS     = 32     # recurrent units
EPOCHS    = 30
SEQ_LENS  = [32, 64, 128]
RUN_EXTRA = True   # set False to skip the additional-exercise cells
VIDEO_SOURCE = "official"   # or "hf_subset" for a faster download
```

## Results summary (single run, CPU)

| Model | Test Accuracy | Macro F1 | Parameters | Training Time |
|---|---|---|---|---|
| RNN | 81.11% | 80.43% | 1,974 | 48.7 s |
| LSTM | 90.89% | 91.13% | 6,006 | 96.3 s |
| GRU | 94.89% | 95.18% | 4,758 | 144.6 s |

Video (CNN–LSTM, selected by validation accuracy): 74.12% test accuracy, 73.02% macro F1.
Seq2Seq (reversal): 100% token accuracy, 100% sequence accuracy.

Full analysis, confusion matrices and discussion answers are in the accompanying lab report.

## Requirements

Runs on Colab's default environment: `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `opencv-python`. All are pre-installed on Colab.

## References

- Anguita et al., *A Public Domain Dataset for Human Activity Recognition Using Smartphones*, ESANN 2013.
- Soomro, Zamir & Shah, *UCF101: A Dataset of 101 Human Action Classes*, 2012.
- Hochreiter & Schmidhuber, *Long Short-Term Memory*, 1997.
