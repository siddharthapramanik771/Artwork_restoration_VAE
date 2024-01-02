# VAE-based Image Restoration Model

## Overview

This repository contains a Variational Autoencoder (VAE)-based image restoration model implemented in PyTorch. The model incorporates skip connections in the decoder to enhance information flow during the upsampling process.
for detailed explanation , see in [medium](https://medium.com/@siddharthapramanik771/image-restoration-using-deep-learning-variational-autoencoders-8483135bb72d).
The used dataset is a smaller version of this dataset from [kaggle](https://www.kaggle.com/datasets/sankarmechengg/art-images-clear-and-distorted)

## Model Architecture

The model consists of an encoder and a decoder, forming the VAE architecture:

### Encoder
- Input: RGB image (3 channels)
- Convolutional layers:
  - Conv1: 3 -> 32 channels, kernel size 4x4, stride 2, padding 1
  - Conv2: 32 -> 64 channels, kernel size 4x4, stride 2, padding 1
  - Conv3: 64 -> 128 channels, kernel size 4x4, stride 2, padding 1
  - Conv4: 128 -> 256 channels, kernel size 4x4, stride 2, padding 1
- Fully connected layers:
  - FC1: 256*8*8 -> 512
  - FC21: 512 -> 128 (mean)
  - FC22: 512 -> 128 (log variance)

### Decoder
- Input: Latent vector of size 128
- Fully connected layers:
  - FC1: 128 -> 512
  - FC2: 512 -> 256*8*8
- Transposed convolutional layers:
  - Conv1: 256 -> 128 channels, kernel size 4x4, stride 2, padding 1
  - Conv2: 128 -> 64 channels, kernel size 4x4, stride 2, padding 1
  - Conv3: 64 -> 32 channels, kernel size 4x4, stride 2, padding 1
  - Conv4: 32 -> 3 channels (RGB), kernel size 4x4, stride 2, padding 1


### Variational Autoencoder (VAE)
- Combines the Encoder and Decoder with skip connections
- Includes a reparameterization step for sampling from the learned distribution

## Benefits of Skip Connections

1. **Information Preservation:**
   - Retains important features from earlier encoder layers for fine detail preservation.

2. **Addressing Vanishing Gradient:**
   - Mitigates the vanishing gradient problem during backpropagation.

3. **Enhanced Image Reconstruction:**
   - Combining low-level and high-level features leads to more accurate reconstructions.

4. **Robustness to Image Distortions:**
   - Helps handle various levels of image distortions by providing additional contextual information.

## Training

The model is trained using distorted and clear image pairs. The loss function includes a binary cross-entropy term and a KL divergence term, allowing the VAE to learn a latent representation of the input images.

Training is performed in two stages:
1. **Clear Image Dataset Training (Stage 1):**
   - The model is trained on a dataset of clear images.
   - The trained model is saved for future use.

2. **Distorted Image Dataset Training (Stage 2):**
   - The model is fine-tuned on a dataset of distorted images.
   - The trained model is saved for future use.

## Evaluation

The model's performance is evaluated on a test dataset. For each test sample, the input image, reconstructed image, and clear image are displayed. Additionally, several image similarity metrics are computed, including Structural Similarity Index (SSI), Mean Squared Error (MSE), and Histogram Correlation.

## Usage

To train the model, run the provided script with the appropriate dataset paths. After training, the model can be loaded for evaluation or further usage.

```python
# Example usage
from vae_model import VAE, load_model

# Load the pre-trained model
vae = VAE()
load_model()
```

## Author
Siddhartha Pramanik

Feel free to reach out for any questions or contributions.