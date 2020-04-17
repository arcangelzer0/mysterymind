########################################################################

import tensorflow as tf
import os

########################################################################
#Directory Helper Function

def create_savefolder(base_directory):
        
    # Add folder name to the checkpoint-dir.
    checkpoint_dir = base_directory #os.path.join(base_directory, "training")
        
    # Create the checkpoint directory if it does not exist.
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        
    return checkpoint_dir

########################################################################
#Network Layers Class

class NetFunctions:
    
    def __init__(self, alias):
        
        # String to distinguish by name
        self.alias = alias
    
    def conv_layer(self,
                   input,              # The layer input
                   num_filters,        # Number of filters
                   filter_size,        # Width and height of each filter
                   conv_pad='SAME',    # Convolution padding
                   conv_stride=1,      # Convolution stride
                   use_pooling=False,  # Use max-pooling
                   pool_size=2,        # Size of pooling kernel
                   pool_pad='SAME',    # Pooling padding
                   use_bn=False,       # Use Batch Normalization?
                   train_mode=None,    # Training phase (True) or test phase (false)?
                   init=None):         # Weight initializer
    
        # Weight initializer for the network
        if init == None: init = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)
        
        # Create tensorflow convolution operation
        layer = tf.layers.conv2d(inputs=input, 
                                 filters=num_filters, kernel_size=filter_size, 
                                 strides=conv_stride, padding=conv_pad,
                                 kernel_initializer=init)
    
        # Use Batch Norm?
        if use_bn:
            layer = tf.layers.batch_normalization(layer, training=train_mode)
    
        # Rectified Linear Unit (ReLU)
        layer = tf.keras.activations.relu(layer)

        # Use pooling to down-sample the image resolution
        if use_pooling:
            layer = tf.layers.max_pooling2d(layer, 
                                            pool_size=(pool_size, pool_size), 
                                            strides=(pool_size, pool_size),
                                            padding=pool_pad)
        return layer
    
    def flatten(self, input):
        
        # layer_shape == [num_images, img_height, img_width, num_channels]
        layer_shape = input.get_shape()

        # The number of features is: img_height * img_width * num_channels
        num_features = layer_shape[1:4].num_elements()
    
        # Reshape the layer to [num_images, num_features].
        layer_flat = tf.reshape(input, [-1, num_features])
        # Shape => [num_images, img_height * img_width * num_channels]

        # Return flattened layer
        return layer_flat
    
    def dense_layer(self,
                    input,              # The layer input
                    num_units,          # Number of neuron units
                    use_relu=True,      # Use Rectified Linear Unit (ReLU)?
                    use_drop=False,     # Use Dropout?
                    drop_prob=0.50,     # Dropout Drop Probability
                    train_mode=None,    # Training phase (True) or test phase (false)?
                    init=None):         # Weight initializer
        
        # Weight initializer for the network
        if init == None: init = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)
        
        # Use ReLU?
        activation = None
        
        if use_relu:
            activation = tf.nn.relu
            
        # Create fully connected layer   
        layer = tf.layers.dense(inputs=input, units=num_units, 
                                activation=activation, 
                                kernel_initializer=init)
        
        # Apply dropout layer?
        if use_drop:
            layer = tf.layers.dropout(inputs=layer, rate=drop_prob, training=train_mode)
        
        return layer

