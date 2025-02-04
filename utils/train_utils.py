import tensorflow as tf
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping # type: ignore
import numpy as np
from datasets.data_loader import dataset_generator


def accuracy(y_true, y_pred):
    y_true = tf.round(y_true)
    y_pred = tf.round(y_pred)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

def train(X_train, y_train, X_val, y_val, site, model, 
                  window_size, epochs, noise_augmentation=False, lr=1e-3, batch_size=64):
    
    # Use the dataset generator to create the training and validation datasets
    train_ds = dataset_generator(X_train, y_train, batch_size=batch_size, training=True, 
                                 noise_augmentation=noise_augmentation)
    val_ds = dataset_generator(X_val, y_val, batch_size=batch_size, training=False)

    # Optimizer
    total_steps = len(train_ds) * epochs
    cosine_lr_decay = tf.keras.optimizers.schedules.CosineDecay(lr, total_steps, alpha=1e-5)
    opt = Adam(learning_rate=cosine_lr_decay)
    
    # Compile the model
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[accuracy])
    if val_ds is None:
        monitor = "loss"
    else:
        monitor = "val_loss"
   
    # Callbacks
    early_stopping = EarlyStopping(monitor=monitor, patience=20, restore_best_weights=True)
    csv_logger = CSVLogger(f"checkpoints/{model.name}_model/{site}{'_noise' if noise_augmentation else ''}_{window_size}.csv", append=True)  
    callbacks = [csv_logger, early_stopping]

    # Train the model
    history = model.fit(train_ds, steps_per_epoch=len(train_ds), epochs=epochs, 
                            validation_data=val_ds, validation_steps=len(val_ds), verbose=1, callbacks=callbacks)
    
    print(history)

    # Save the model
    tf.keras.models.save_model(model, f"checkpoints/{model.name}_model/{site}{'_noise' if noise_augmentation else ''}_{window_size}.h5")
    
    return model
