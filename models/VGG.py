import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import Dropout, MaxPool1D, Flatten, Dense

def VGG(window_size, F, dropout_rate, nb_block=3):
    inputs = Input(shape=(window_size, 21))
    x = inputs
    nb_filter = F

    for i in range(nb_block):
        x = Conv1D(filters=nb_filter, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

        x = Conv1D(filters=nb_filter, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool1D(pool_size=2, padding='same')(x)
        x = Dropout(dropout_rate)(x)
        
        nb_filter *= 2

    x = Flatten()(x)
    x = Dense(units=1)(x)
    outputs = Activation('sigmoid')(x)

    # Create model
    VGG_model = Model(inputs=inputs, outputs=outputs, name='VGG')
    VGG_model.summary()
    
    # Save model architecture
    tf.keras.utils.plot_model(VGG_model, 
                              to_file=f'checkpoints/VGG_model/{window_size}.png', show_shapes=True)

    return VGG_model

