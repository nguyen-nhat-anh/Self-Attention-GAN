import tensorflow as tf
from tensorflow.keras import layers
from ops import SNConv2D, SNDense, SelfAttention


D_CONV_DIM = 64


class InitDownResBlock(layers.Layer):
    def __init__(self, channels_out, **kwargs):
        super(InitDownResBlock, self).__init__(**kwargs)
        self.channels_out = channels_out
        
        
    def build(self, input_shape):
        '''
        input_shape = (batch_size, height, width, channels)
        '''
        self.conv1 = SNConv2D(self.channels_out, (3, 3), strides=(1, 1), padding='same', name='conv1')
        self.conv2 = SNConv2D(self.channels_out, (3, 3), strides=(1, 1), padding='same', name='conv2')
        self.conv3 = SNConv2D(self.channels_out, (1, 1), strides=(1, 1), padding='valid', name='conv3')
        super(InitDownResBlock, self).build(input_shape)  # Be sure to call this at the end
    
    
    def call(self, inputs, training=None):
        '''
        inputs - shape = (batch_size, height, width, channels)
        '''
        x = self.conv1(inputs, training=training) # (batch_size, height, width, channels_out)
        x = layers.LeakyReLU(0.2)(x)
        
        x = self.conv2(x, training=training) # (batch_size, height, width, channels_out)
        x = layers.AveragePooling2D(padding='same')(x) # (batch_size, height/2, width/2, channels_out)

        x0 = layers.AveragePooling2D(padding='same')(inputs) # (batch_size, height/2, width/2, channels)
        x0 = self.conv3(x0, training=training) # (batch_size, height/2, width/2, channels_out)
        return x + x0
    
    
class DownResBlock(layers.Layer):
    def __init__(self, channels_out, downsample=True, **kwargs):
        super(DownResBlock, self).__init__(**kwargs)
        self.channels_out = channels_out
        self.downsample = downsample
        
        
    def build(self, input_shape):
        '''
        input_shape = (batch_size, height, width, channels)
        '''
        self.channels_in = input_shape.as_list()[-1]
        self.channels_mismatch = (self.channels_in != self.channels_out)
        
        self.conv1 = SNConv2D(self.channels_out, (3, 3), strides=(1, 1), padding='same', name='conv1')
        self.conv2 = SNConv2D(self.channels_out, (3, 3), strides=(1, 1), padding='same', name='conv2')
        self.conv0 = SNConv2D(self.channels_out, (1, 1), strides=(1, 1), padding='valid', name='conv0')
        super(DownResBlock, self).build(input_shape)  # Be sure to call this at the end
    
    
    def call(self, inputs, training=None):
        '''
        inputs - shape = (batch_size, height, width, channels)
        '''
        x = layers.LeakyReLU(0.2)(inputs)
        x = self.conv1(x, training=training) # (batch_size, height, width, channels_out)
        
        x = layers.LeakyReLU(0.2)(x)
        x = self.conv2(x, training=training) # (batch_size, height, width, channels_out)
        if self.downsample:
            x = layers.AveragePooling2D(padding='same')(x) # (batch_size, height/2, width/2, channels_out)
        
        x0 = inputs # (batch_size, height, width, channels)
        if self.downsample or self.channels_mismatch:
            x0 = self.conv0(x0, training=training) # (batch_size, height, width, channels_out)
            if self.downsample:
                x0 = layers.AveragePooling2D(padding='same')(x0) # (batch_size, height/2, width/2, channels_out)
        return x + x0 # (batch_size, height/2, width/2, channels_out) if downsample=True
                      # (batch_size, height, width, channels_out) if downsample=False
        
        
def make_discriminator_model():
    input_layer = layers.Input(shape=(128, 128, 3), name='disc_input') # (None, 128, 128, 3)
    x = InitDownResBlock(D_CONV_DIM, name='init_disc_block')(input_layer) # (None, 64, 64, D_CONV_DIM)
    
    x = DownResBlock(D_CONV_DIM*2, name='disc_block1')(x) # (None, 32, 32, D_CONV_DIM*2)
    
    x = SelfAttention(name='disc_attention')(x) # (None, 32, 32, D_CONV_DIM*2)
    
    x = DownResBlock(D_CONV_DIM*4, name='disc_block2')(x) # (None, 16, 16, D_CONV_DIM*4)
    
    x = DownResBlock(D_CONV_DIM*8, name='disc_block3')(x) # (None, 8, 8, D_CONV_DIM*8)
    
    x = DownResBlock(D_CONV_DIM*16, name='disc_block4')(x) # (None, 4, 4, D_CONV_DIM*16)
    
    x = DownResBlock(D_CONV_DIM*16, downsample=False, name='disc_block5')(x) # (None, 4, 4, D_CONV_DIM*16)
    
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=[1, 2]), name='disc_global_sum2d')(x) # (None, D_CONV_DIM*16)
    
    output_layer = SNDense(1, name='disc_last_fc')(x) # (None, 1)

    return tf.keras.Model(input_layer, output_layer)