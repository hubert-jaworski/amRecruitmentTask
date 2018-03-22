import pickle
import numpy as np
import matplotlib.pyplot as plt
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