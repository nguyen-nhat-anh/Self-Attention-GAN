import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from glob import glob

from generator import make_generator_model
from discriminator import make_discriminator_model
from ops import TrainArg

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # set visible gpus
##############################
# Constants
##############################
DATA_DIR = os.path.join('data', 'celebA')
IMG_PATH_PATTERN = os.path.join(DATA_DIR, '*.*')
CKPT_DIR = os.path.join('model', 'celebA')

N_SAMPLES = len(glob(IMG_PATH_PATTERN))
BUFFER_SIZE = 5000
BATCH_SIZE = 50

NOISE_DIM = 128
NUM_EXAMPLES_TO_GENERATE = 25
SAMPLE_DIR = os.path.join('samples', 'celebA')
# SEED = tf.random.normal(shape=(NUM_EXAMPLES_TO_GENERATE, 1, 1, NOISE_DIM))
SEED = tf.random.truncated_normal(shape=(NUM_EXAMPLES_TO_GENERATE, 1, 1, NOISE_DIM), stddev=0.5)

# D_LEARNING_RATE = 4e-04
# G_LEARNING_RATE = 1e-04
D_LEARNING_RATE = 2e-04
G_LEARNING_RATE = 5e-05
BETA_1 = 0.0
# BETA_2 = 0.9
BETA_2 = 0.999
ADAM_EPSILON = 1e-08

START_ITER = 0
END_ITER = 200000
DISPLAY_FREQUENCY = 500
SAVE_FREQUENCY = 5000


##############################
# Prepare dataset
##############################
def decode_img(img_path):
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, [128, 128]) # resize image
    img = (tf.cast(img, tf.float32) - 127.5) / 127.5 # Normalize the images to [-1, 1]
    return img


dataset = tf.data.Dataset.list_files(IMG_PATH_PATTERN)
dataset = dataset.map(decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat().prefetch(tf.data.experimental.AUTOTUNE)

iterator = dataset.make_one_shot_iterator()
image_ph = iterator.get_next()


##############################
# Create models and optimizers
##############################
generator = make_generator_model(NOISE_DIM)
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(G_LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=ADAM_EPSILON)
discriminator_optimizer = tf.keras.optimizers.Adam(D_LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=ADAM_EPSILON)


##############################
# Train step
##############################
hinge_loss = tf.keras.losses.Hinge()
def discriminator_real_loss(real_output):
    return hinge_loss(tf.ones_like(real_output), real_output)

def discriminator_fake_loss(fake_output):
    return hinge_loss(-tf.ones_like(fake_output), fake_output)

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def train_step(inputs):
    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
#         noise = tf.random.normal(shape=(BATCH_SIZE, 1, 1, NOISE_DIM))
        noise = tf.random.truncated_normal(shape=(BATCH_SIZE, 1, 1, NOISE_DIM), stddev=0.5)
        real_output = discriminator(inputs, training=TrainArg.TRUE_UPDATE_U)
        fake_output = discriminator(generator(noise, training=TrainArg.TRUE_UPDATE_U), training=TrainArg.TRUE_NO_UPDATE_U)
        
        disc_loss_op = discriminator_real_loss(real_output) + discriminator_fake_loss(fake_output)
        gen_loss_op = generator_loss(fake_output)

    disc_grads = disc_tape.gradient(disc_loss_op, discriminator.trainable_variables)
    disc_train_op = discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    update_ops = generator.get_updates_for(generator.inputs) # to update moving_mean and moving_variance of BatchNormalization layers
    disc_train_op = tf.group([disc_train_op, update_ops])
    
    # make sure `loss`es will only be returned after `train_op`s have executed
    with tf.control_dependencies([disc_train_op]): 
        disc_loss_op_id = tf.identity(disc_loss_op)
    
    gen_grads = gen_tape.gradient(gen_loss_op, generator.trainable_variables)
    gen_train_op = generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    gen_train_op = tf.group([gen_train_op, update_ops])
    
    # make sure `loss`es will only be returned after `train_op`s have executed
    with tf.control_dependencies([gen_train_op]): 
        gen_loss_op_id = tf.identity(gen_loss_op)
        
    return disc_loss_op_id, gen_loss_op_id
    
##############################
# Training loop
##############################

def generate_and_save_images(sess, index):
    test_fake_image = generator(SEED, training=TrainArg.FALSE)
    predictions = sess.run(test_fake_image)
    fig = plt.figure(figsize=(8,8))

    for i in range(NUM_EXAMPLES_TO_GENERATE):
        plt.subplot(5, 5, i+1)
        plt.imshow((predictions[i] * 127.5 + 127.5).astype(int))
        plt.axis('off')

    plt.savefig(os.path.join(SAMPLE_DIR, 'iter-{}.jpg'.format(str(index))))


mean_disc_loss_op, mean_gen_loss_op = train_step(image_ph)
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=2)

with tf.Session() as sess:
    sess.run(init)

    latest_ckpt = tf.train.latest_checkpoint(CKPT_DIR)
    if latest_ckpt is None:
        print('No checkpoint found!')
    else:
        print('Found checkpoint : "{}"'.format(latest_ckpt))
        saver.restore(sess, latest_ckpt)
        START_ITER = int(latest_ckpt.split('-')[-1])

    for iteration in range(START_ITER, END_ITER):
        start = time.time()
        
        noise = np.random.normal(size=(BATCH_SIZE, 1, 1, NOISE_DIM))
        disc_loss = sess.run(mean_disc_loss_op)
        gen_loss = sess.run(mean_gen_loss_op)

        if (iteration + 1) % DISPLAY_FREQUENCY == 0:
            generate_and_save_images(sess, iteration + 1)

        print('discriminator loss: {}'.format(disc_loss))
        print('generator loss: {}'.format(gen_loss))
        print('Time for iteration {}/{} is {} sec'.format(iteration + 1, END_ITER, time.time()-start))
        print('#############################################')

        if (iteration + 1) % SAVE_FREQUENCY == 0:
            saver.save(sess, os.path.join(CKPT_DIR, 'model'), global_step=iteration + 1)