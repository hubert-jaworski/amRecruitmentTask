import numpy as np
import pickle
from collections import Counter

from keras.callbacks import ReduceLROnPlateau

from utils import load_data
from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import rmsprop, adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

def calculate_class_weight(labels_dict):
    m = max(labels_dict.values())
    class_weights = dict()

    for label in labels_dict.keys():
        if labels_dict[label]:
            score = m/labels_dict[label]
        else:
            score = 1
        class_weights[label] = score
    return class_weights


def train_classifier(train_images, train_labels, validation_images, validation_labels, class_weights):
    classifier = Sequential()
    classifier.add(Conv2D(filters=32, kernel_size=(7, 7), padding='Same', activation='relu', input_shape=(56, 56, 1)))
    classifier.add(Conv2D(filters=32, kernel_size=(7, 7), padding='Same', activation='relu'))
    classifier.add(MaxPool2D(pool_size=(4, 4), strides=(4, 4)))
    classifier.add(Dropout(0.25))

    classifier.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    classifier.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    classifier.add(Dropout(0.25))

    classifier.add(Flatten())
    classifier.add(Dense(256, activation="relu"))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(36, activation="softmax"))

    optimizer = rmsprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    classifier.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    data_generator = ImageDataGenerator(rotation_range=10,  zoom_range=0.10,
                                        width_shift_range=0.10, height_shift_range=0.10)


    data_generator.fit(train_images)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    classifier.fit_generator(data_generator.flow(train_images, train_labels, batch_size=32),
                             validation_data=(validation_images, validation_labels),
                             epochs=10, callbacks=[learning_rate_reduction])


    return classifier

def do_cross_validation(images, labels):
    folds = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=2018).split(images, labels))
    val_scores = []
    for fold_id, (train_idx, validation_idx) in enumerate(folds):
        train_images = images[train_idx]
        train_labels = labels[train_idx]
        validation_images = images[validation_idx]
        validation_labels = labels[validation_idx]

        classifier = train_classifier(train_images, train_labels, validation_images, validation_labels, class_weights)

        val_scores.append(classifier.evaluate(validation_images, validation_labels, batch_size=32))

    return val_scores

if __name__ == '__main__':
    images, labels = load_data('train_sample.pkl')
    images = images.reshape(-1, 56, 56, 1)
    class_weights = calculate_class_weight(Counter(labels))

    labels = to_categorical(labels, num_classes=36)

    images_tst, labels_tst = load_data('test_sample.pkl')
    images_tst = images_tst.reshape(-1, 56, 56, 1)
    labels_tst = to_categorical(labels_tst, num_classes=36)


    classifier = train_classifier(images, labels, images_tst, labels_tst, class_weights)
    score = classifier.evaluate(images_tst, labels_tst, batch_size=32)

    classifier.save('keras_cnn_model.h5')