# Artwork_restoration_VAE
an Variational Autoencoder model to restore the artwork images 

The Variational AutoEncoder (VAE) in this implementation follows the typical architecture of a generative model, containing two primary components : an Encoder and a Decoder.

1. **Encoder:** The role of the Encoder is to translate the high dimensional input data (images) into a lower-dimensional format representing latent variables. Here is the breakdown of the implemented Encoder:
    - `conv1`: First convolutional layer that takes an image with 3 channels (RGB) and applies 32 filters of size 4x4, Striding at a pace of 2, and with padding of 1.
    - `conv2`: The second layer takes 32 channels as input and applies 64 filters of 4x4 size.
    - `conv3`: The third layer consumes 64 channels and applies 128 filters of size 4x4.
    - `conv4`: The fourth layer receives 128 channels as input and applies 256 filters of the 4x4 size.
    - `fc1`: Followed by a fully connected layer that flattens the output of the last conv layer and reduces its dimensionality to 512.
    - `fc21` & `fc22`: The last two layers are both fully connected. They generate parameters of the latent distribution, particularly the mean (mu) and variance (logvar).
2. **Decoder:** The Decoder's role is to translate the lower-dimensional latent space back into the original input image dimensions. Here is the breakdown of the implemented Decoder:
    - `fc1`: First fully connected layer that takes the latent vector (dimension 128) as input and outputs a tensor of size 512
    - `fc2`: The second layer scales up the output dimension to 256 * 8 * 8
    - `conv1`: First transpose convolution layer (also known as deconvolution), accepting 256 channels, applying 128 filters of 4x4, with stride 2 and padding 1
    - `conv2`: Second transpose convolution layer, taking 128 channels, applying 64 filters of 4x4.
    - `conv3`: Third transpose convolution layer with 64 input channels and 32 output filters of 4x4 size.
    - `conv4`: Fourth transpose convolution layer taking 32 channels, applying 3 filters. The number of output filters corresponds to the number of color channels of the output image (RGB)

Apart from the Encoder and Decoder, two other important components are:

1. **Reparameterization:** VAE training requires a stochastic sampling process. To backpropagate gradients through this stochastic process, the 'reparameterization trick' is used. This Decoder's output re-samples in epsilon from a standard normal distribution and re-scales and shifts it using the mean and variance obtained from the Encoder's output.
2. **Loss Function:** The VAE is trained with a loss function composed of two parts: Binary Cross Entropy (BCE) and the Kullback-Leibler Divergence (KLD). The BCE measures how well the Decoder has reconstructed the original image, while the KLD measures how much the latent distribution deviates from a unit Gaussian.

Training the VAE involves an alternation between using the Decoder to generate images, comparing the generated images to the original via the loss function, and then using backpropagation to update the weights of the Decoder and Encoder to minimize the loss.
