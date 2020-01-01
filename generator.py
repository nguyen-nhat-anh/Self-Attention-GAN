import tensorflow as tf
from tensorflow.keras import layers
from ops import SNConv2D, SNDense, SelfAttention


NOISE_DIM = 128
G_CONV_DIM = 64


class BatchNorm(layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        
    
    def build(self, input_shape):
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-05, name='batch_normalization')
        super(BatchNorm, self).build(input_shape)
        
    def call(self, inputs, training=None):
        x = self.bn(inputs, training=bool(training))
        return x


class UpResBlock(layers.Layer):
    def __init__(self, channels_out, **kwargs):
        super(UpResBlock, self).__init__(**kwargs)
        self.channels_out = channels_out
    
    def build(self, input_shape):
        '''
        input_shape = (batch_size, height, width, channels)
        '''
        self.deconv1 = SNConv2D(self.channels_out, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='deconv1')
        self.deconv2 = SNConv2D(self.channels_out, (3, 3), strides=(1, 1), padding='same', name='deconv2')
        self.deconv3 = SNConv2D(self.channels_out, (1, 1), strides=(1, 1), padding='valid', use_bias=False, name='deconv3')
        self.bn1 = BatchNorm(name='bn1')
        self.bn2 = BatchNorm(name='bn2')
        super(UpResBlock, self).build(input_shape)  # Be sure to call this at the end
                
    def call(self, inputs, training=None):
        '''
        inputs - shape = (batch_size, height, width, channels)
        '''
        x = self.bn1(inputs, training=training)
        x = layers.ReLU()(x)
        x = layers.UpSampling2D()(x) # (batch_size, 2*height, 2*width, channels)
        x = self.deconv1(x, training=training) # (batch_size, 2*height, 2*width, channels_out)

        x = self.bn2(x, training=training)
        x = layers.ReLU()(x)
        x = self.deconv2(x, training=training) # (batch_size, 2*height, 2*width, channels_out)

        x0 = layers.UpSampling2D()(inputs) # (batch_size, 2*height, 2*width, channels)
        x0 = self.deconv3(x0, training=training) # (batch_size, 2*height, 2*width, channels_out)
        return x + x0
    
    
def make_generator_model():
    input_layer = layers.Input(shape=(1, 1, NOISE_DIM), name='gen_input') # (None, 1, 1, NOISE_DIM)
    x = SNDense(4*4*(G_CONV_DIM*16), name='gen_first_fc')(input_layer) # (None, 1, 1, 4*4*G_CONV_DIM*16)
    x = layers.Reshape((4, 4, G_CONV_DIM*16))(x) # (None, 4, 4, G_CONV_DIM*16)

    x = UpResBlock(G_CONV_DIM*16, name='gen_block1')(x) # (None, 8, 8, G_CONV_DIM*16)

    x = UpResBlock(G_CONV_DIM*8, name='gen_block2')(x) # (None, 16, 16, G_CONV_DIM*8)

    x = UpResBlock(G_CONV_DIM*4, name='gen_block3')(x) # (None, 32, 32, G_CONV_DIM*4)

    x = SelfAttention(name='gen_attention')(x) # (None, 32, 32, G_CONV_DIM*4)

    x = UpResBlock(G_CONV_DIM*2, name='gen_block4')(x) # (None, 64, 64, G_CONV_DIM*2)

    x = UpResBlock(G_CONV_DIM, name='gen_block5')(x) # (None, 128, 128, G_CONV_DIM)

    x = BatchNorm(name='gen_bn')(x)

    x = layers.ReLU()(x)

    x = SNConv2D(3, (3, 3), strides=(1, 1), padding='same', name='gen_last_conv')(x) # (None, 128, 128, 3)
    output_layer = layers.Activation('tanh')(x)

    return tf.keras.Model(input_layer, output_layer)