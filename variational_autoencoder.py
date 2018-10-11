'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
# from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model, np_utils
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
# from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
# from keras.models import load_model

# from PIL import Image
# from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
import ioFunction_version_4_3 as IO

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import keras

# for tensorboard
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import glob

old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)


# reparameterization trick
## instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    # epsilon.eval(session = K.random_normal(shape=(batch, dim)))
    print("ε=", epsilon, "(≧▽≦)")
    return z_mean + K.exp(0.5 * z_log_var) * epsilon



def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_3D"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join("result/",model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1]
                # , c=y_test
                )
    # plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    # plt.show()

    filename = os.path.join("result/", model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 15
    digit_size = 27
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            plt.imshow(digit, cmap='Greys_r')
            # plt.savefig(str(i) + '@' + str(j) + 'fig.png')
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|raw'):
#     return [os.path.join(root, f)
#             for root, _, files in os.walk(directory) for f in files
#             if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

image_size = 9
original_dim = image_size * image_size * image_size

# load train data
print('load data')
x_train = np.zeros((10000, original_dim))
y_train = np.zeros((10000, original_dim))

with open('E:/git/VAE/linear/1/train/filename.txt', 'rt') as f:
    i = 0
    for line in f:
        if i >= 10000:
            break
        line = line.split()
        x_train[i, :] = IO.read_raw(line[0], dtype='double')

        i += 1

    print(x_train.shape)

# load test data
print('load data')
x_test = np.zeros((1000, original_dim))
y_test = np.zeros((1000, original_dim))

with open('E:/git/VAE/linear/1/val/filename.txt', 'rt') as f:
    i = 0
    for line in f:
        if i >= 1000:
            break
        line = line.split()
        x_test[i, :] = IO.read_raw(line[0], dtype='double')

        i += 1

    print(x_test.shape)

# file_list = glob.glob('/linear/1/*')
#
# # load image data
# x = []
# y = []
# # for picture in list_pictures('./linear/1/'):
# #     img = IO.read_mhd_and_raw(picture)
# for filename in file_list:
#     with open(filename, 'r') as f:
#         print(f)
#         input = IO.read_mhd_and_raw(f)
#         x.append(input)
#         y.append(0)
#
# x = np.asarray(x)
# y = np.asarray(y)
#
# x = x.astype('double')
# x = x / 255.0
y_train = np_utils.to_categorical(y_train, 1)
y_test = np_utils.to_categorical(y_test, 1)
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=111)

# x_train = np.reshape(x_train, [-1, original_dim])
# x_test = np.reshape(x_test, [-1, original_dim])

# MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# image_size = x_train.shape[1]
# original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('double')
x_test = x_test.astype('double')

scaler = MinMaxScaler()
MinMaxScaler(copy=True, feature_range=(0, 1))
scaler.fit(x_train,x_test)
x_train = scaler.transform(x_train)
x_test = scaler.fit_transform(x_test)

# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 100

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='result/vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='result/vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

# generator
# d_h = Dense(intermediate_dim, activation='relu')
# d_out = Dense(original_dim, activation='sigmoid')
#
# generator_in = Input(shape=(2,))
# generator_h = d_h(generator_in)
# generator_out = d_out(generator_h)
#
# generator = Model(generator_in, generator_out)
# generator.summary()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='result/vae_mlp.png',
               show_shapes=True)

    ### add for TensorBoard
    tb_cb = keras.callbacks.TensorBoard(log_dir="~/tflog/", histogram_freq=0, write_graph=True)
    cbks = [tb_cb]
    ###

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        fit = vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                      callbacks=cbks,
                      validation_data=(x_test, None)
                      # , verbose=0
                      )

        vae.save_weights('vae_mlp_3D.h5')

    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_mlp")

    # 画像の出力
    s = 0
    n = 10
    plt.figure(figsize=(5, 3.1))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(n):

        hidden_imgs = encoder.predict(x_test)
        model_imgs = decoder.predict(hidden_imgs[0])
        vae_imgs = vae.predict(x_test)
        IO.write_raw(vae_imgs[i + s].reshape(1,729), "E:/git/VAE/result/"+'vae_'+str(i)+'.raw')
        # original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i + s].reshape(27, 27))
        plt.axis('off')
        plt.gray()

        # reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(model_imgs[i + s].reshape(27, 27))
        plt.axis('off')

        # vae model
        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(vae_imgs[i + s].reshape(27, 27))
        plt.axis('off')



    plt.show()

    # ----------------------------------------------
    # Some plots
    # ----------------------------------------------
    fig, ax = plt.subplots(ncols=1, figsize=(10,4))

    # loss
    # def plot_history_loss(fit):
    #     # Plot the loss in the history
    #     ax.plot(fit.history['loss'],label="loss for training")
    #     ax.plot(fit.history['val_loss'],label="loss for validation")
    #     ax.set_title('model loss')
    #     ax.set_xlabel('epoch')
    #     ax.set_ylabel('loss')
    #     ax.legend(loc='upper right')
    #
    # plot_history_loss(fit)
    # fig.savefig('./vae.png')
    # plt.close()


    def plot_history(history, path):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['loss', 'val_loss'], loc='lower right')
        plt.savefig(os.path.join(path, 'loss.jpg'))
        plt.show()

    plot_history(history=fit, path="E:/git/VAE/result/")

### add for TensorBoard
KTF.set_session(old_session)
###

