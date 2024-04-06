import tensorflow as tf
import numpy as np



def resblock(inputs, out_channel=32, name='resblock'):
    
    with tf.variable_scope(name):
        
        x = tf.keras.layers.Conv2D(out_channel, (3, 3), activation=None, name='conv1')(inputs)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(out_channel, (3, 3), activation=None, name='conv2')(x)
        
        return x + inputs




def unet_generator(inputs, channel=32, num_blocks=4, name='generator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        
        x0 = tf.keras.layers.Conv2D(channel, (7, 7), activation=None)(inputs)
        x0 = tf.nn.relu(x0)
        
        x1 = tf.keras.layers.Conv2D(channel, (3, 3), strides=2, activation=None)(x0)
        x1 = tf.nn.relu(x1)
        x1 = tf.keras.layers.Conv2D(channel*2, (3, 3), activation=None)(x1)
        x1 = tf.nn.relu(x1)
        
        x2 = tf.keras.layers.Conv2D(channel*2, (3, 3), strides=2, activation=None)(x1)
        x2 = tf.nn.relu(x2)
        x2 = tf.keras.layers.Conv2D(channel*4, (3, 3), activation=None)(x2)
        x2 = tf.nn.relu(x2)
        
        for idx in range(num_blocks):
            x2 = resblock(x2, out_channel=channel*4, name='block_{}'.format(idx))
            
        x2 = tf.keras.layers.Conv2D(channel*2, (3, 3), activation=None)(x2)
        x2 = tf.nn.relu(x2)
        
        h1, w1 = tf.shape(x2)[1], tf.shape(x2)[2]
        x3 = tf.image.resize(x2, (h1*2, w1*2))
        x3 = tf.keras.layers.Conv2D(channel*2, (3, 3), activation=None)(x3+x1)
        x3 = tf.nn.relu(x3)
        x3 = tf.keras.layers.Conv2D(channel, (3, 3), activation=None)(x3)
        x3 = tf.nn.relu(x3)

        h2, w2 = tf.shape(x3)[1], tf.shape(x3)[2]
        x4 = tf.image.resize(x3, (h2*2, w2*2))
        x4 = tf.keras.layers.Conv2D(channel, (3, 3), activation=None)(x4+x0)
        x4 = tf.nn.relu(x4)
        x4 = tf.keras.layers.Conv2D(3, (7, 7), activation=None)(x4)
        
        return x4

if __name__ == '__main__':
    

    pass
