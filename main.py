import numpy as np
from sklearn.utils import shuffle
import Network
import pickle

# Reference
# https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e
# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

if __name__ == '__main__':

    # Load data
    x_train_raw = np.load('data/train_inputs.npy')
    y_train_raw = np.load('data/train_targets.npy')

    x_valid_raw = np.load('data/valid_inputs.npy')
    y_valid_raw = np.load('data/valid_targets.npy')

    x_test_raw = np.load('data/test_inputs.npy')
    y_test_raw = np.load('data/test_targets.npy')

    vocab = np.load('data/vocab.npy')

    # Shuffle train data set
    x_train, y_train = shuffle(x_train_raw, y_train_raw, random_state=0)
    x_val, y_val = shuffle(x_valid_raw, y_valid_raw, random_state=0)

    # Construct model
    model = Network.Network()

    x_train = model.get_one_hot(x_train)
    x_val = model.get_one_hot(x_val)

    pkl_filename = "model.pk"

    # Parameter sets
    learning_rate = [0.012]  # 0.012 # 0.015
    learning_rate_decay = [0.95]
    reg = 0.00001
    batch_size = [900]  # 800 # 850 # 900
    epochs = 5
    num_iter = 75  # 75 # 50

    parameters = list((x, y, z)
                      for x in learning_rate
                      for y in learning_rate_decay
                      for z in batch_size)

    # Use SGD to optimize the parameters in the model
    num_train = x_train.shape[0]

    loss_history = []
    train_acc_history = []
    val_acc_history = []
    best_acc = 0

    for lr, lrd, bs in parameters:
        print('Parameter set: learning rate: {},'
              ' learning rate decay: {},'
              ' batch size: {},'
              ' iter num for learning rate decay: {}'.format(lr, lrd, bs, num_iter))
        iterations_per_epoch = max(num_train // bs, 1)
        print('Iterations per epoch: {}'.format(iterations_per_epoch))
        for epoch in range(epochs):
            print('Training epoch: {}'.format(epoch + 1))
            x_train, y_train = shuffle(x_train, y_train, random_state=0)

            for iteration in range(0, iterations_per_epoch):
                start = iteration * bs
                end = (iteration + 1) * bs

                x_batch, y_batch = x_train[start:end], y_train[start:end]

                # Compute loss and gradients using the current minibatch
                loss, grads = model.backward_propagation(x_batch, y_batch, reg=reg)
                loss_history.append(loss)

                # Update the parameters
                for param in ['W2', 'b1']:
                    for i in range(0, 3):
                        model.params[param][i] -= lr * grads[param][i]

                for param in ['W3', 'W1', 'b2']:
                    model.params[param] -= lr * grads[param]

                # Decay learning rate
                if (iteration + 1) % num_iter == 0:
                    lr *= lrd

                # Every epoch, check train and val accuracy and decay learning rate.
                if (iteration+1) % iterations_per_epoch == 0:
                    train_acc = (model.predict(x_batch) == y_batch).mean()
                    val_acc = (model.predict(x_val) == y_val).mean()
                    print('train_acc: %f / validation_acc: %f' % (train_acc, val_acc))
                    train_acc_history.append(train_acc)
                    val_acc_history.append(val_acc)

        # Choose the model with highest accuracy (parameter setting selection)
        if val_acc_history[-1] > best_acc:
            best_acc = val_acc_history[-1]
            best_train_acc = train_acc_history[-1]
            best_model = model
            best_params = (lr, lrd, bs)

    print('Best validation accuracy is %f' % best_acc)

    with open(pkl_filename, 'wb') as file:
        pickle.dump(best_model, file)

    print('Training is completed!')

