import itertools
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def load_data(path):
    f = open(path, 'rb')
    images, labels = pickle.load(f)
    f.close()
    return images, labels

def preprocess_data(raw_images, raw_labels):
    images = raw_images
    labels = raw_labels.flatten()

    images = images[labels != 30]
    labels = labels[labels != 30]

    return images, labels

def make_train_test_split(images, labels, test_size):
    images_tr, images_tst, labels_tr, labels_tst = train_test_split(images, labels, test_size=test_size,
                                                                    random_state=2018)

    test_sample = (images_tst, labels_tst)
    with open('test_sample.pkl', 'wb') as f:
        pickle.dump(test_sample, f)

    train_sample = (images_tr, labels_tr)
    with open('train_sample.pkl', 'wb') as f:
        pickle.dump(train_sample, f)


def plot_img(img, label):
    plt.imshow(np.reshape(img, (56, 56)), cmap='binary')
    plt.suptitle(label)
    plt.show()

def plot_img_for_each_class(images, labels):
    for label in set(labels):
        images_of_label = images[labels==label]
        for img in images_of_label[:3]:
            plot_img(img, label)


def calculate_class_weight(labels):
    labels_dict = Counter(labels)
    m = max(labels_dict.values())
    class_weights = dict()

    for label in labels_dict.keys():
        if labels_dict[label]:
            score = m / labels_dict[label]
        else:
            score = 1
        class_weights[label] = score
    return class_weights

def plot_confusion_matrix(labels_true, labels_predicted):
    classes = ['6', 'P', 'O', 'V', 'W', '3', 'A', '8', 'T', '1', '0', '9', 'H', 'R', 'N', '7', 'K', 'L', 'G', '4', 'Y',
               'C', 'E', 'J', '5', 'I', 'S', '2', 'F', 'Z', '-', 'Q', 'M', 'B', 'D', 'U']

    cm = confusion_matrix(labels_true, labels_predicted, labels=range(36))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

if __name__ == '__main__':
    from predict import predict

    images_tst, labels_tst = load_data('test_sample.pkl')
    images_tst = images_tst.reshape(-1, 56, 56, 1)
    labels_tst = to_categorical(labels_tst, num_classes=36)

    labels_predicted = predict(images_tst)
    labels_true = np.argmax(labels_tst, axis=1)

    plot_confusion_matrix(labels_true, labels_predicted)