import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(file_name, site, window_size, empty_aa = '*'):
    half_len = (window_size - 1) // 2
    
    # Load raw_data, only keep rows with 'code' column equal to site
    raw_data = pd.read_csv(file_name)
    raw_data = raw_data[raw_data['code'] == site]
    
    # Create 'short_sequences' centered at the 'position' with a length equal to 'window_size', pad with empty_aa
    def get_short_sequence(row):
        sequence = row['sequence']
        position = row['position'] - 1  # Convert to 0-based index
        start = max(0, position - half_len)
        end = min(len(sequence), position + half_len + 1)
        short_seq = sequence[start:end]
        if start == 0:
            short_seq = empty_aa * (half_len - position) + short_seq
        if end == len(sequence):
            short_seq = short_seq + empty_aa * (half_len - (len(sequence) - position - 1))            
        return short_seq

    raw_data['short_sequence'] = raw_data.apply(get_short_sequence, axis=1)
    
    # Split short_sequence into individual characters
    short_sequences_split = raw_data['short_sequence'].apply(lambda x: pd.Series(list(x)))
    
    # Use pd.get_dummies to one-hot encode the characters
    short_sequences_encoded = pd.get_dummies(short_sequences_split, dtype='int')
    
    # Ensure the columns are ordered correctly according to the letters
    letters = "ACDEFGHIKLMNPQRSTVWY*"
    columns = [f"{i}_{letter}" for i in range(window_size) for letter in letters]
    short_sequences_encoded = short_sequences_encoded.reindex(columns=columns, fill_value=0)

    # Convert the DataFrame to a numpy array
    X = short_sequences_encoded.values.reshape((len(raw_data), window_size, -1))

    # Get the labels
    y = raw_data['label'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #Split the training data into training and validating sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Save the data
    np.savez(f'datasets/{site}_{window_size}.npz', X_train=X_train, X_test=X_test, X_val=X_val, y_train=y_train, y_test=y_test, y_val=y_val)

    return

if __name__ == '__main__':
    sites = ['ST', 'Y']
    window_sizes = [15, 33, 51]
    for site in sites:
        for window_size in window_sizes:
            get_data('datasets/raw_data.csv', site, window_size)








