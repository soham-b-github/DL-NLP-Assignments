# Autoencoder Assignment
## Date: 28th March 2025
1) Implement an Autoencoder by adding Bernoulli dropout noise as one of the layers into the encoder module. Use a CNN architecture.

2) Implement an Sparse Autoencoder using KL Regularisation.


The dataset to be used is Fashion MNIST Dataset which has 10 classes.
Training samples are 60K
Test samples are 10K
image size is 28x28
Attached is a sample code of a Vanilla MLP in Keras.
In the code, the training dataset has been split into a train set (54,000 images) and a validation set (6000 images). Please retain the same split when reporting your results.


Experiment with hyperparameters such as number of neurons, layers, batch size, activation functions and report the best validation loss for both the autoencoders along with the respective configuration details.

Perturb the latent vectors of sample 10 images by adding small noise, reconstruct those images and plot them to see how they appear.

You can run your code either in Keras/Pytorch however working in Pytorch would be helpful for job opportunities.
