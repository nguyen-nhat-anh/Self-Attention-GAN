import tensorflow as tf
from tensorflow.keras import layers
from enum import Enum


# WEIGHT_INIT = tf.contrib.layers.xavier_initializer()
WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
BIAS_INIT = tf.zeros_initializer()


class TrainArg(Enum):
    FALSE = 0
    TRUE_UPDATE_U = 1
    TRUE_NO_UPDATE_U = 2
    
    
def _l2normalize(v, eps=1e-12):
    """l2 normize the input vector."""
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm(weights, u, num_iters=1, update_collection=None, with_sigma=False):
    w_shape = weights.shape.as_list()
    w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
    u_ = u
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))

    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /= sigma
    if update_collection is None:
        with tf.control_dependencies([u.assign(u_)]):
            w_bar = tf.reshape(w_mat, w_shape)
    else:
        w_bar = tf.reshape(w_mat, w_shape)
        if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_))
    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar
    
    
class SNConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, use_bias=True,
                 kernel_initializer=WEIGHT_INIT, bias_initializer=BIAS_INIT, **kwargs):
        super(SNConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        
    
    def build(self, input_shape):
        '''
        input_shape = (batch_size, height, width, channels_in)
        '''
        self.channels_in = input_shape.as_list()[-1]
        self.kernel = self.add_weight("kernel",
                                      shape=(self.kernel_size[0], self.kernel_size[1], self.channels_in, self.filters), 
                                      initializer=self.kernel_initializer)
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=(self.filters,), initializer=self.bias_initializer)
        self.u = self.add_weight("u", shape=(1, self.kernel.shape.as_list()[-1]), initializer=tf.truncated_normal_initializer(), trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        super(SNConv2D, self).build(input_shape)  # Be sure to call this at the end
    
    
    def call(self, inputs, training=None):
        if training == TrainArg.TRUE_UPDATE_U:
            x = tf.nn.conv2d(inputs, spectral_norm(self.kernel, self.u, update_collection=None), self.strides, self.padding)
        else:
            x = tf.nn.conv2d(inputs, spectral_norm(self.kernel, self.u, update_collection='NO_OPS'), self.strides, self.padding)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x
    
    
class SNDense(layers.Layer):
    def __init__(self, units, use_bias=True,
                 kernel_initializer=WEIGHT_INIT, bias_initializer=BIAS_INIT, **kwargs):
        super(SNDense, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        
    
    def build(self, input_shape):
        '''
        input_shape = (batch_size, ..., input_dim)
        '''
        self.input_dim = input_shape.as_list()[-1]
        self.kernel = self.add_weight("kernel", 
                                      shape=(self.input_dim, self.units), 
                                      initializer=self.kernel_initializer)
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=(self.units,), initializer=self.bias_initializer)
        self.u = self.add_weight("u", shape=(1, self.kernel.shape.as_list()[-1]), initializer=tf.truncated_normal_initializer(), trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        super(SNDense, self).build(input_shape)  # Be sure to call this at the end
    
    
    def call(self, inputs, training=None):
        if training == TrainArg.TRUE_UPDATE_U:
            x = tf.matmul(inputs, spectral_norm(self.kernel, self.u, update_collection=None))
        else:
            x = tf.matmul(inputs, spectral_norm(self.kernel, self.u, update_collection='NO_OPS'))
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x
    
    
class SelfAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        
        
    def build(self, input_shape):
        '''
        input_shape = (batch_size, height, width, channels)
        '''
        _, self.height, self.width, self.channels_in = input_shape.as_list()
        self.conv_f = SNConv2D(self.channels_in // 8, (1, 1), strides=(1, 1), padding='valid', name='conv_f')
        self.conv_g = SNConv2D(self.channels_in // 8, (1, 1), strides=(1, 1), padding='valid', name='conv_g')
        self.conv_h = SNConv2D(self.channels_in // 2, (1, 1), strides=(1, 1), padding='valid', name='conv_h')
        self.conv_v = SNConv2D(self.channels_in, (1, 1), strides=(1, 1), padding='valid', name='conv_v')
        self.gamma = self.add_weight('gamma', shape=(1,), initializer=tf.constant_initializer(0.0))
        super(SelfAttention, self).build(input_shape)  # Be sure to call this at the end
    
    
    def call(self, inputs, training=None):
        '''
        inputs - shape = (batch_size, height, width, channels)
        '''
        f = self.conv_f(inputs, training=training) # (batch_size, height, width, channels/8)
        f = layers.MaxPool2D(padding='SAME')(f) # (batch_size, height/2, width/2, channels/8)
        f = layers.Reshape((self.height * self.width // 4, self.channels_in // 8))(f) # (batch_size, height*width/4, channels/8)
        
        g = self.conv_g(inputs, training=training) # (batch_size, height, width, channels/8)
        g = layers.Reshape((self.height * self.width, self.channels_in // 8))(g) # (batch_size, height*width, channels/8)
        
        h = self.conv_h(inputs, training=training) # (batch_size, height, width, channels/2)
        h = layers.MaxPool2D(padding='SAME')(h) # (batch_size, height/2, width/2, channels/2)
        h = layers.Reshape((self.height * self.width // 4, self.channels_in // 2))(h) # (batch_size, height*width/4, channels/2)
        
        s = tf.matmul(g, f, transpose_b=True) # (batch_size, height*width, height*width/4)
        beta = tf.nn.softmax(s, axis=-1) # (batch_size, height*width, height*width/4)
        
        o = tf.matmul(beta, h) # (batch_size, height*width, channels/2)
        o = layers.Reshape((self.height, self.width, self.channels_in // 2))(o) # (batch_size, height, width, channels/2)
        o = self.conv_v(o, training=training) # (batch_size, height, width, channels)
        
        return self.gamma * o + inputs # (batch_size, height, width, channels)