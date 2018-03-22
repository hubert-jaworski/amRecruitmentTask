import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from utils import load_data, preprocess_data, make_train_test_split, plot_img_for_each_class

# raw_images, raw_labels = preprocess_data(*load_data('train.pkl'))
# make_train_test_split(raw_images, raw_labels, 0.2)


# Let's use sklearn's MultiLayerPerceptrion Classifier.
# The classifier has hyperparameters to be tuned.
# We'll use grid search with 4-fold Cross Validation (multithreading on 4 CPU cores) for the following parameters:
# 1) Number of hidden layers (1, 2, 3, 4, 5)
# 2) Number of neurons in each layer (100, 200, 300, 400)
#     note: number of neurons may differ between layers, here constant for simplicity
# 3) Regularization parameter (1e-2, 1e-1, 1, 10)
#
# The rest of the parameters will be set to constant, arbitrarily chosen values:
# Activation function for hidden layers: tanh
# Weight optimization solver: adam
# Adam optimizer parameters:
#     beta_0 = 0.9
#     beta_1 = 0.999
#     eps = 1e-8
# Batch size: 200
# Shuffle samples?: True
# Learning rate function: constant
# Initial learning rate: 1e-3

# Other hyperparameters, not mentioned above are:
# Classifier model itself (e.g. MLP vs SVN)
# Class balancing: method + parameters (e.g. upsampling minority classes, threshold number of samples)
# Loss Function (cross-entropy, possibly with higher weights for minority classes)
# (Nesterov's) momentum rate - only for sgd solver

def check_hyperpars_on_grid(images, labels, hidden_layers=(1,), neurons=(50,), regularization=(1e-2,)):
    hyperparameters_scores = []
    for number_of_hidden_layers in hidden_layers:
        for number_of_neurons in neurons:
            for regularization in regularization:
                hidden_layers = tuple(number_of_neurons for layer in range(number_of_hidden_layers))

                classifier = MLPClassifier(hidden_layer_sizes=hidden_layers, activation='tanh', solver='adam',
                                           alpha=regularization, batch_size=200, learning_rate_init=0.001, max_iter=200,
                                           shuffle=True, random_state=2018, verbose=False,
                                           beta_1=0.9, beta_2=0.999, epsilon=1e-8)

                score = cross_val_score(classifier, images, labels, cv=4, n_jobs=-1)
                print('layers: {}, neurons: {}, reg: {}, score: {}'.format(number_of_hidden_layers, number_of_neurons,
                                                                           regularization, np.mean(score)))
                hyperparameters_scores.append(
                    (number_of_hidden_layers, number_of_neurons, regularization, np.mean(score)))

    return hyperparameters_scores

if __name__ == '__main__':
    images, labels = load_data('train_sample.pkl')
    # plot_img_for_each_class(images, labels)
    #
    # hyperparameters_scores = check_hyperpars_on_grid(images, labels, hidden_layers=(1, 2, 3, 4, 5),
    #                                                  neurons=(100, 200, 300, 400), regularization=(1e-2, 1e-1, 1, 10))
    #
    # hyperparameters_scores = sorted(hyperparameters_scores, key=lambda tup: tup[3], reverse=True)
    # for tuple in hyperparameters_scores:
    #     print(tuple)

    classifier = MLPClassifier(hidden_layer_sizes=(200, 200, 200), activation='tanh', solver='adam',
                               alpha=1, batch_size=200, learning_rate_init=0.001, max_iter=200,
                               shuffle=True, random_state=2018, verbose=True,
                               beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    classifier.fit(images, labels)

    images_tst, labels_tst = load_data('test_sample.pkl')
    print('Test score: {0:.2f}'.format(100*classifier.score(images_tst, labels_tst)))

    with open('sklearn_mlpc_model.pkl', 'wb') as f:
        pickle.dump(classifier, f)