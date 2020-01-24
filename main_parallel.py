import argparse
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from glob import glob

from generator import make_generator_model
from discriminator import make_discriminator_model
from ops import TrainArg


##############################
# Constants
##############################
N_GPUS = 3

DATASET_NAME = 'celebA'

BUFFER_SIZE = 5000
BATCH_SIZE_PER_REPLICA = 50

NOISE_DIM = 128
NUM_EXAMPLES_TO_GENERATE = 25

D_LEARNING_RATE = 2e-04 # 4e-04
G_LEARNING_RATE = 5e-05 # 1e-04
BETA_1 = 0.0
BETA_2 = 0.999 # 0.9
ADAM_EPSILON = 1e-08

START_ITER = 0
END_ITER = 200000
DISPLAY_FREQUENCY = 500
SAVE_FREQUENCY = 5000


##############################
# Command-line argument parser
##############################
argparser = argparse.ArgumentParser(description="Self-Attention GAN for multiple gpus")

argparser.add_argument('--dataset_name', type=str, default=DATASET_NAME)
argparser.add_argument('--batch_size_per_replica', type=int, default=BATCH_SIZE_PER_REPLICA, help='batch size per gpu')
argparser.add_argument('--n_gpus', type=int, default=N_GPUS, help='number of gpus to use')

argparser.add_argument('--d_learning_rate', type=float, default=D_LEARNING_RATE, help='discriminator learning rate')
argparser.add_argument('--g_learning_rate', type=float, default=G_LEARNING_RATE, help='generator learning rate')
argparser.add_argument('--beta_1', type=float, default=BETA_1, help='Adam optimizer beta_1')
argparser.add_argument('--beta_2', type=float, default=BETA_2, help='Adam optimizer beta_2')
argparser.add_argument('--adam_epsilon', type=float, default=ADAM_EPSILON, help='Adam optimizer epsilon')

argparser.add_argument('--n_iters', type=int, default=END_ITER, help='number of training iterations')
argparser.add_argument('--display_frequency', type=int, default=DISPLAY_FREQUENCY, help='how often sample images are generated')
argparser.add_argument('--save_frequency', type=int, default=SAVE_FREQUENCY, help='how often the model is saved')

def _main(args):
    ##############################
    # Distribute strategy
    ##############################
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) > 0, "No GPUs available."
    tf.config.experimental.set_visible_devices(gpus[-args.n_gpus:], 'GPU') # use only the last n gpus
    strategy = tf.distribute.MirroredStrategy() # create a mirrored strategy for data parallel
    
    
    ##############################
    # Constants
    ##############################
    DATA_DIR = os.path.join('data', args.dataset_name)
    IMG_PATH_PATTERN = os.path.join(DATA_DIR, '*.*')
    CKPT_DIR = os.path.join('model', args.dataset_name + '_parallel')
    
    N_SAMPLES = len(glob(IMG_PATH_PATTERN))
    GLOBAL_BATCH_SIZE = args.batch_size_per_replica * strategy.num_replicas_in_sync

    SAMPLE_DIR = os.path.join('samples', args.dataset_name + '_parallel')
#     SEED = tf.random.normal(shape=(NUM_EXAMPLES_TO_GENERATE, 1, 1, NOISE_DIM))
    SEED = tf.random.truncated_normal(shape=(NUM_EXAMPLES_TO_GENERATE, 1, 1, NOISE_DIM), stddev=0.5)
    
    
    ##############################
    # Create directories
    ##############################
    assert os.path.exists(DATA_DIR)
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
    if not os.path.exists(SAMPLE_DIR):
        os.makedirs(SAMPLE_DIR)
        
    
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
    dataset = dataset.shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE, drop_remainder=True).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    with strategy.scope():
        iterator = dist_dataset.make_initializable_iterator()
        iterator_init = iterator.initialize()
        image_ph = iterator.get_next()


    ##############################
    # Create models and optimizers
    ##############################
    with strategy.scope():
        generator = make_generator_model(NOISE_DIM)
        discriminator = make_discriminator_model()

        generator_optimizer = tf.keras.optimizers.Adam(args.g_learning_rate, beta_1=args.beta_1, 
                                                       beta_2=args.beta_2, epsilon=args.adam_epsilon)
        discriminator_optimizer = tf.keras.optimizers.Adam(args.d_learning_rate, beta_1=args.beta_1, 
                                                           beta_2=args.beta_2, epsilon=args.adam_epsilon)


    ##############################
    # Train step
    ##############################
    with strategy.scope():
        hinge_loss = tf.keras.losses.Hinge(reduction='none')
        def discriminator_real_loss(real_output):
            per_sample_loss = hinge_loss(tf.ones_like(real_output), real_output) # (batch_size_per_replica,)
            loss = tf.reduce_sum(per_sample_loss) / tf.cast(GLOBAL_BATCH_SIZE, per_sample_loss.dtype) # scalar
            return loss

        def discriminator_fake_loss(fake_output):
            per_sample_loss = hinge_loss(-tf.ones_like(fake_output), fake_output) # (batch_size_per_replica,)
            loss = tf.reduce_sum(per_sample_loss) / tf.cast(GLOBAL_BATCH_SIZE, per_sample_loss.dtype) # scalar
            return loss

        def generator_loss(fake_output):
            return -tf.reduce_sum(fake_output) / tf.cast(GLOBAL_BATCH_SIZE, fake_output.dtype) # scalar

        def train_step(inputs):
            '''
            inputs: "per-replica" values, such as those produced by a "distributed Dataset"
            '''
            with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape: 
#                 noise = tf.random.normal(shape=(args.batch_size_per_replica, 1, 1, NOISE_DIM))
                noise = tf.random.truncated_normal(shape=(args.batch_size_per_replica, 1, 1, NOISE_DIM), stddev=0.5)
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
    with strategy.scope():
        def distributed_train_step(dataset_inputs):
            per_replica_disc_losses, per_replica_gen_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
            mean_disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_disc_losses)
            mean_gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_gen_losses)
            return mean_disc_loss, mean_gen_loss


        def generate_and_save_images(sess, index):
            test_fake_image = generator(SEED, training=TrainArg.FALSE)
            predictions = sess.run(test_fake_image)
            fig = plt.figure(figsize=(8,8))

            for i in range(NUM_EXAMPLES_TO_GENERATE):
                plt.subplot(5, 5, i+1)
                plt.imshow((predictions[i] * 127.5 + 127.5).astype(int))
                plt.axis('off')

            plt.savefig(os.path.join(SAMPLE_DIR, 'iter-{}.jpg'.format(str(index))))


        mean_disc_loss_op, mean_gen_loss_op = distributed_train_step(image_ph)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=2)

        with tf.Session() as sess:
            sess.run(init)
            sess.run(iterator_init)

            latest_ckpt = tf.train.latest_checkpoint(CKPT_DIR)
            if latest_ckpt is None:
                print('No checkpoint found!')
            else:
                print('Found checkpoint : "{}"'.format(latest_ckpt))
                saver.restore(sess, latest_ckpt)
                START_ITER = int(latest_ckpt.split('-')[-1])

            for iteration in range(START_ITER, args.n_iters):
                start = time.time()

                disc_loss = sess.run(mean_disc_loss_op)
                gen_loss = sess.run(mean_gen_loss_op)

                if (iteration + 1) % args.display_frequency == 0:
                    generate_and_save_images(sess, iteration + 1)

                print('discriminator loss: {}'.format(disc_loss))
                print('generator loss: {}'.format(gen_loss))
                print('Time for iteration {}/{} is {} sec'.format(iteration + 1, args.n_iters, time.time()-start))
                print('#############################################')

                if (iteration + 1) % args.save_frequency == 0:
                    saver.save(sess, os.path.join(CKPT_DIR, 'model'), global_step=iteration + 1)

                
if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)