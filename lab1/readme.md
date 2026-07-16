# Single Layer Perceptron for Banknote Authentication

## Aim

To implement a Single Layer Perceptron from scratch and use it to classify banknotes as authentic or forged, based on statistical features extracted from wavelet-transformed images of the notes.

## Dataset

**Banknote Authentication Dataset** — UCI Machine Learning Repository
[https://archive.ics.uci.edu/dataset/267/banknote+authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

| Property | Value |
|---|---|
| Instances | 1372 |
| Features | 4 (numerical) |
| Classes | 2 (binary) |
| Missing values | None |

**Features:**
- `variance` — variance of the wavelet-transformed image
- `skewness` — skewness of the wavelet-transformed image
- `curtosis` — kurtosis of the wavelet-transformed image
- `entropy` — entropy of the image

**Target:**
- `0` — Authentic banknote
- `1` — Forged banknote

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

**Dependencies:**
- pandas
- numpy
- matplotlib
- scikit-learn

## Running the Project

The entire implementation — data loading, EDA, preprocessing, the perceptron, training, evaluation, and all additional experiments — is contained in a single Jupyter notebook: `lab_1_perceptron.ipynb`.

To run it:

```bash
jupyter notebook lab_1_perceptron.ipynb
```

Ensure `data_banknote_authentication.txt` is placed in the same directory as the notebook, then run all cells in order from top to bottom.

## Conclusion

A Single Layer Perceptron was implemented from scratch and trained on the Banknote Authentication dataset over 50 epochs at a learning rate of 0.01. The model achieved a test accuracy of 98.18%, with precision, recall, and F1-score all above 0.96, including a perfect recall of 1.0 — meaning every forged note in the test set was correctly identified.

The training error did not converge to zero and instead oscillated between roughly 12 and 20 misclassifications in later epochs, indicating that the two classes are approximately, but not perfectly, linearly separable. This was consistent with the overlap observed in the scatter plot and decision boundary visualization. A comparison across learning rates (0.001, 0.01, 0.1) produced identical results, since the perceptron's step activation depends only on the sign of the weighted sum and not its magnitude. Cross-checking against scikit-learn's built-in `Perceptron` gave comparable performance, confirming the correctness of the from-scratch implementation.

Overall, the experiment demonstrated both the capability and the limitations of a single artificial neuron, motivating the use of multilayer architectures for problems that are not linearly separable.
