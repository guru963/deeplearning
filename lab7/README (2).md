# MNIST Variational Autoencoder (VAE)

A deep-learning lab project that explores how an Autoencoder and a
Variational Autoencoder (VAE) learn compressed representations of
handwritten digit images from the MNIST dataset.

## Overview

A VAE learns a compact, probabilistic latent representation of each
input image. The decoder uses a latent vector to reconstruct an image.
Unlike a basic Autoencoder, a VAE learns the mean (`mu`) and variance
(commonly represented through `log_var`) of a latent distribution, then
samples a latent vector during training.

This project focuses on: - Loading and preprocessing MNIST digit
images. - Building and training a VAE. - Reconstructing input images. -
Visualising the two-dimensional latent space. - Plotting latent means
(`mu`) for test images, coloured by digit label. - Evaluating
reconstruction quality with metrics such as MSE, MAE and SSIM, if
implemented in the notebook/script.

## How a VAE Works

1.  **Encoder:** Receives an image and predicts latent distribution
    parameters, usually `mu` and `log_var`.
2.  **Reparameterisation:** Samples a latent vector using the predicted
    distribution while allowing gradients to flow during training.
3.  **Decoder:** Uses the latent vector to reconstruct the input image.
4.  **Training objective:** Combines reconstruction loss with
    KL-divergence regularisation.

A common objective is:

`Total Loss = Reconstruction Loss + beta * KL Divergence`

The reconstruction term encourages the output to resemble the input. The
KL term encourages the encoded distributions to stay close to a standard
normal prior.

## Latent-Space Visualisation

For a two-dimensional latent space, each test image can be represented
by a point `(z1, z2)`. In an encoder-mean plot, the coordinates are the
two components of the encoder's mean vector `mu`.

-   Each point represents one test image.
-   The point's colour represents the image's true digit label (0--9),
    used for visualisation.
-   Overlapping colours indicate that different digit classes have
    overlapping latent representations.
-   The VAE is generally trained for reconstruction and latent
    regularisation, not to create perfectly separated digit classes.

## Requirements

Python 3.10 or newer is recommended. Install the dependencies from
`requirements.txt`:

``` bash
pip install -r requirements.txt
```

The requirements file uses TensorFlow/Keras, NumPy, Matplotlib,
scikit-image, scikit-learn, pandas and Pillow. Keep package versions
compatible with your Python version and environment.

## Running the Project

1.  Clone or download this project.

2.  Create and activate a virtual environment (recommended).

3.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

4.  Open the project notebook or run the Python script containing the
    VAE implementation.

5.  Run the cells or script in order. The first run may download the
    MNIST dataset, depending on how the dataset loader is configured.

> **Note:** Update this section with the exact notebook or script
> filename used in your repository. The filename was not provided when
> this README was created.

## Expected Outputs

Depending on what is implemented in the notebook/script, outputs may
include: - Original MNIST test images. - Reconstructed images produced
by the decoder. - Training and validation loss plots. - A
two-dimensional latent-space scatter plot. - A latent-space plot using
encoder means (`mu`) with points coloured by digit label. -
Reconstruction metrics such as MSE, MAE or SSIM, if included in the
implementation.

Actual results depend on the model architecture, hyperparameters, random
seed and training environment.

## Key Concepts

  -----------------------------------------------------------------------
  Concept                             Meaning
  ----------------------------------- -----------------------------------
  Encoder                             Maps an input image to latent
                                      distribution parameters.

  Latent space                        Compact numerical representation
                                      learned by the model.

  `mu`                                Mean of the encoder's latent
                                      distribution.

  `log_var`                           Log-variance parameter commonly
                                      predicted by the encoder.

  Reparameterisation                  Samples a latent vector in a way
                                      that supports gradient-based
                                      training.

  Decoder                             Reconstructs an image from a latent
                                      vector.

  Reconstruction loss                 Measures the difference between
                                      input and reconstructed image.

  KL divergence                       Regularises the latent distribution
                                      towards the chosen prior.
  -----------------------------------------------------------------------

## Troubleshooting

-   **TensorFlow installation fails:** Check your Python version and
    install a TensorFlow version compatible with your operating system
    and Python environment.
-   **Notebook cannot find a package:** Activate the intended virtual
    environment, then run `pip install -r requirements.txt`.
-   **Latent classes overlap:** This can be normal for a VAE; the
    objective does not directly require the ten digit labels to form
    separate clusters.
-   **Reconstructions look blurry:** This can occur with VAEs and
    depends on the decoder, loss function, latent dimension and training
    setup.

## Project Scope

This README describes a general MNIST VAE workflow. Adjust the commands,
filenames, architecture details and reported metrics to match the exact
code in your repository.

## License

Add the license used by your project here.
