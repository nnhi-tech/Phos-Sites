import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import Dropout, Flatten, Dense, Add
from tensorflow.keras import backend as K 

def res_blocks(x, nb_filter, stride):
    shortcut = x
    x = Conv1D(filters=nb_filter, kernel_size=3, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=nb_filter, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Check if the shape of x and shortcut are the same
    if K.int_shape(x) != K.int_shape(shortcut):
        shortcut = Conv1D(filters=nb_filter, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x


def ResNet(window_size, F, dropout_rate, nb_block=3):
    inputs = Input(shape=(window_size, 21))
    x = inputs  
    stride = 1
    nb_filter = F

    x = Conv1D(filters=nb_filter, kernel_size=3, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for i in range(nb_block):
        x = res_blocks(x, nb_filter, stride) 
        x = Dropout(dropout_rate)(x)
        x = res_blocks(x, nb_filter, stride) 
        
        stride = 2
        nb_filter *= 2
  
    x = Flatten()(x)
    x = Dense(units=1)(x)
    outputs = Activation('sigmoid')(x)

    # Create model
    ResNet_model = Model(inputs=inputs, outputs=outputs, name='ResNet')
    ResNet_model.summary()
    
    # Save model architecture
    tf.keras.utils.plot_model(ResNet_model, 
                              to_file=f'checkpoints/ResNet_model/{window_size}.png', show_shapes=True)

    return ResNet_model
