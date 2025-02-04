import numpy as np
from models.VGG import VGG
from models.ResNet import ResNet
from models.CRNN import CRNN
from utils.train_utils import train
from utils.test_utils import test

def main():     
    sites = ['ST', 'Y']
    models = ['VGG', 'ResNet', 'LSTM', 'GRU']
    inital_Fs = [32, 16, 32, 32]
    dropout_rates = [0.1, 0.1, 0.05, 0.05]
    window_sizes = [15, 33, 51]

    for site in sites:
        for model_name, F, dropout_rate in zip(models, inital_Fs, dropout_rates):
            for window_size in window_sizes:

                # Load model
                if model_name == 'VGG':
                    model = VGG(window_size, F, dropout_rate)
                elif model_name == 'ResNet':
                    model = ResNet(window_size, F, dropout_rate)
                elif model_name == 'LSTM':
                    model = CRNN(window_size, F, dropout_rate, rnn_type='LSTM')    
                elif model_name == 'GRU':
                    model = CRNN(window_size, F, dropout_rate, rnn_type='GRU')

                # Load data
                data = np.load(f'datasets/{site}_{window_size}.npz')  
                X_train, y_train = data['X_train'], data['y_train']
                X_test, y_test = data['X_test'], data['y_test']
                X_val, y_val = data['X_val'], data['y_val']

                # Define epochs
                if site == 'Y':
                    epochs = 1
                elif site == 'ST':
                    epochs = 1

                # Train and test without noise augmentation
                train(X_train, y_train, X_val, y_val,
                                site, model, window_size, epochs)
                test(X_test, y_test, site, model_name, window_size)

                # Train and test with noise augmentation
                train(X_train, y_train, X_val, y_val,
                                site, model, window_size, epochs, noise_augmentation=True)
                test(X_test, y_test, site, model_name, window_size, noise_augmentation=True)
                   
    return

if __name__ == '__main__':
    main()