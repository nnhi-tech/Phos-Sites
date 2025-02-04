import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
from datasets.data_loader import dataset_generator
import os
import pandas as pd
import numpy as np

def test(X_test, y_test, site, model_name, 
                  window_size, noise_augmentation=False):
    
    # Load the model
    model_path = f"checkpoints/{model_name}_model/{site}{'_noise' if noise_augmentation else ''}_{window_size}.h5"
    model = tf.keras.models.load_model(model_path)

    # Generate the test dataset
    test_ds = dataset_generator(X_test, y_test, batch_size=32, training=False)
    
    y_true = []
    y_pred = []
    
    for X_batch, y_batch in test_ds:
        y_true.extend(y_batch.numpy())
        y_pred.extend(model.predict(X_batch))
    
    # Calculate metrics
    y_true = np.array(y_true).ravel()
    y_pred = np.round(np.array(y_pred).ravel())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(model_path)
    print(f'Test accuracy: {acc}')
    print(f'F1 Score: {f1}')
    
    # Save metrics to CSV file
    report_test_file = f'Report_test.csv'
    model_info = f"{model_name}_model_{site}{'_noise' if noise_augmentation else ''}_{window_size}"
    
    metrics_df = pd.DataFrame({
        'model_info': [model_info],
        'acc': [acc],
        'F1': [f1]
    })
        
    # Check if the report file exists and if it contains any data
    if os.path.exists(report_test_file) and os.path.getsize(report_test_file) > 0:
        header = False
    else:
        header = True

    metrics_df.to_csv(report_test_file, mode='a', header=header, index=False)
    
    return 
