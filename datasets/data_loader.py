import tensorflow as tf

def augment_fn(X, y, training=False, noise_augmentation=False):
    # Add noise to input and output
    val_X, val_y = 0.1, 0.4
    if training and noise_augmentation:
        noise_X = tf.random.uniform(shape=tf.shape(X), minval=-val_X, maxval=val_X, dtype=tf.float32)
        noise_y = tf.random.uniform(shape=tf.shape(y), minval=0, maxval=val_y, dtype=tf.float32)
        X = tf.cast(X, tf.float32) + noise_X
        y = tf.cast(y, tf.float32)
        y = tf.where(y == 0., y + noise_y, y - noise_y)
    return X, y

def dataset_generator(inputs, labels, batch_size, training=False, noise_augmentation=False):
    ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
    ds = ds.map(lambda X, y: augment_fn(X, y, training=training, noise_augmentation=noise_augmentation),
                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(len(inputs)).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds



