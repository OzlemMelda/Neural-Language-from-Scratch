import numpy as np
import pickle

if __name__ == '__main__':

    # Load data
    x_test = np.load('data/test_inputs.npy')
    y_test = np.load('data/test_targets.npy')

    # Load trained model
    pkl_filename = "model.pk"
    with open(pkl_filename, 'rb') as file:
        final_model = pickle.load(file)

    x_test = final_model.get_one_hot(x_test)

    # Calculate test accuracy
    test_acc = (final_model.predict(x_test) == y_test).mean()

    print('Test accuracy is %f' % test_acc)

