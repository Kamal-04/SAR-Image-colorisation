# noinspection PyUnresolvedReferences
#Pix2PixGAN
from keras.initializers import RandomNormal
from keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from numpy import ones
from numpy import zeros
from numpy.random import randint
import pandas as pd


images_folder = 'Images/'
generator_folder = 'G_model/'
discriminator_folder = 'D_model/'
loss_file = 'loss_values.xlsx'
loss_plots_folder = 'Loss_plots/'

run_count = 'C4_'

def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)  # As described in the original paper

    # source image input
    in_src_image = Input(shape=image_shape)  # Image we want to convert to another image
    # target image input
    in_target_image = Input(shape=image_shape)  # Image we want to generate after training.

    # concatenate Images, channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
 
    # C64: 4x4 kernel Stride 2x2
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128: 4x4 kernel Stride 2x2
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256: 4x4 kernel Stride 2x2
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512: 4x4 kernel Stride 2x2
    # Not in the original paper. Comment this block if you want.
    # d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    # d = BatchNormalization()(d)
    # d = LeakyReLU(alpha=0.2)(d)
    # second last output layer : 4x4 kernel but Stride 1x1
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02, seed=42)
    # add downsampling layer
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g


# define a decoder block to be used in generator
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02, seed=42)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g


# define the standalone generator model - U-net
def define_generator(image_shape=(256, 256, 3)):
    # weight initialization
    init = RandomNormal(stddev=0.02, seed=42)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model: C64-C128-C256-C512-C512-C512-C512-C512
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    # print(f"Shape of e1: {e1.shape}")
    e2 = define_encoder_block(e1, 128, batchnorm=False)
    # print(f"Shape of e2: {e2.shape}")
    e3 = define_encoder_block(e2, 256)
    # print(f"Shape of e3: {e3.shape}")
    e4 = define_encoder_block(e3, 512)
    # print(f"Shape of e4: {e4.shape}")
    e5 = define_encoder_block(e4, 512)
    # print(f"Shape of e5: {e5.shape}")
    e6 = define_encoder_block(e5, 512)
    # print(f"Shape of e6: {e6.shape}")
    e7 = define_encoder_block(e6, 512)
    # print(f"Shape of e7: {e7.shape}")
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # print(f"Shape of b: {b.shape}")
    # decoder model: CD512-CD512-CD512-C512-C256-C128-C64
    d1 = decoder_block(b, e7, 512)
    # print(f"Shape of d1: {d1.shape}")
    d2 = decoder_block(d1, e6, 512)
    # print(f"Shape of d2: {d2.shape}")
    d3 = decoder_block(d2, e5, 512)
    # print(f"Shape of d3: {d3.shape}")
    d4 = decoder_block(d3, e4, 512, dropout=False)
    # print(f"Shape of d4: {d4.shape}")
    d5 = decoder_block(d4, e3, 256, dropout=False)
    # print(f"Shape of d5: {d5.shape}")
    d6 = decoder_block(d5, e2, 128, dropout=False)
    # print(f"Shape of d6: {d6.shape}")
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # print(f"Shape of d7: {d7.shape}")
    # output
    g = Conv2DTranspose(image_shape[2], (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)  # Modified
    out_image = Activation('tanh')(g)  # Generates Images in the range -1 to 1. So change inputs also to -1 to 1
    # print(f"Shape of g: {g.shape}")
    # define model
    model = Model(in_image, out_image)
    return model


def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False  # Descriminator layers set to untrainable in the combined GAN but
            # standalone descriminator will be trainable.

    # define the source image
    in_src = Input(shape=image_shape)
    # supply the image as input to the generator
    gen_out = g_model(in_src)
    print(f"Generator output shape: {gen_out.shape}")
    # supply the input image and generated image as inputs to the discriminator
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and disc. output as outputs
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)

    # Total loss is the weighted sum of adversarial loss (BCE) and L1 loss (MAE)
    # Authors suggested weighting BCE vs L1 as 1:100.
    model.compile(loss=['binary_crossentropy', 'mae'],
                  optimizer=opt, loss_weights=[5, 100])
    return model


# select a batch of random samples, returns Images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected Images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


# generate a batch of Images, returns Images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def summarize_performance(step, g_model, d_model, dataset, n_samples=3, ):
    # select a sample of input Images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0

    # plt.title("D_LOSS: " + (str((d_loss1 + d_loss2) / 2)) + "G_LOSS" + str(g_loss))
    # plot real source Images
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.title("SRC")
        plt.imshow(X_realA[i], cmap="gray")
    # plot generated target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.title("GEN")
        plt.imshow(X_fakeB[i], cmap="gray")
    # plot real target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples * 2 + i)
        plt.axis('off')
        plt.title("TAR")
        plt.imshow(X_realB[i], cmap="gray")
    # save plot to file
    images_folder_ = images_folder + run_count + '_plot_%09d.png' % (step + 1)
    plt.savefig(images_folder_)
    plt.close()
    # save the generator model
    generator_folder_ = generator_folder + run_count + '_256g_%09d.keras' % (step + 1)
    g_model.save(generator_folder_)
    discriminator_folder_ = discriminator_folder + run_count + '_256d_%09d.keras' % (step + 1)
    d_model.save(discriminator_folder_)
    print('>Saved: %s, %s and %s' % (images_folder_, generator_folder_, discriminator_folder_))


def saving_loss(g_loss, d_loss_1, d_loss_2, steps):
    # Append new loss data to the CSV
    new_data_ = pd.DataFrame([[g_loss, d_loss_1, d_loss_2]], columns=['G_loss', 'D_loss_1', 'D_loss_2'])
    new_data_.to_csv(loss_file, mode='a', index=False, header=False)

    # Read the entire loss data for plotting
    data_ = pd.read_csv(loss_file)

    plt.figure(figsize=(10, 6))  # Optional: Set the figure size

    # Plot each column against the index
    for col in data_.columns:
        plt.plot(data_.index, data_[col], label=col)  # Explicitly plot against the index

    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Plot of Losses')
    plt.legend()
    plt.grid(True)

    # Save the plot
    file_name_ = loss_plots_folder + run_count + '_loss_plot_%09d.png' % (steps + 1)  # Added file extension .png
    plt.savefig(file_name_)
    plt.close()

    print('> %s plot successfully saved' % file_name_)


def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i + 1) % bat_per_epo == 0:
            summarize_performance(i, g_model, d_model, dataset, n_samples=3)

        if (i + 1) % (bat_per_epo//10)== 0:
            saving_loss(g_loss, d_loss1, d_loss2, i)
