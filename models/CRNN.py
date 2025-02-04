import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import Dropout, MaxPool1D, LSTM, GRU

def CRNN(window_size, F, dropout_rate, rnn_type, nb_block=2):
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

    # Add RNN layer
    if rnn_type == 'LSTM':
        x = LSTM(units=2*F, return_sequences=True)(x)
        x = LSTM(units=1, return_sequences=True)(x)

    elif rnn_type == 'GRU':
        x = GRU(units=2*F, return_sequences=True)(x)
        x = GRU(units=1, return_sequences=True)(x)

    # Max pooling
    x = tf.math.reduce_max(x, axis=1)

    outputs = Activation('sigmoid')(x)

    # Create model
    CRNN_model = Model(inputs=inputs, outputs=outputs, name=rnn_type)
    CRNN_model.summary()
    
    # Save model architecture
    tf.keras.utils.plot_model(CRNN_model, 
                              to_file=f'checkpoints/{rnn_type}_model/{window_size}.png', show_shapes=True)

    return CRNN_model

