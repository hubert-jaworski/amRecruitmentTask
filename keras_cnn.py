from copy import copy

import numpy as np
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.optimizers import rmsprop, adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

from utils import load_data, calculate_class_weight

HP_SPACE = {
            'filters_1': [16, 32, 48],
            'filters_2': [16, 32, 48],
            'kernel_1': [5, 7, 9],
            'kernel_2': [3, 5, 7, 9],
            'pooling_1': [5, 6, 7],
            'pooling_2': [1, 2, 3],
            'beta_1': [1 - (10 ** (-p)) for p in range(1, 3)],
            'beta_2': [1 - (10 ** (-p)) for p in range(2, 5)],
            'eps': [10 ** (-p) for p in range(10, 13)],
            'learning_rate': [10 ** (-p) for p in [2.5, 3, 3.5, 4]],
            'rho': [1 - (10 ** (-p)) for p in [1, 1.5, 2]],
            'decay': [10 ** (-p) for p in range(1, 4)],
            'hidden_units': [64, 128, 192, 256, 320],
            'activation': ['relu', 'tanh'],
            'use_img_augmentation': [False, True],
            'use_adam': [False, True],
            'use_dropout': [False, True],
            'use_class_weight': [False, True],
            'use_lr_reduction': [False, True],
        }

class HyperparametersHandler():
    def __init__(self, initial_parameters=None):
        self.space = HP_SPACE
        self.current = initial_parameters or {key: np.random.randint(0, len(self.space[key])) for key in self.space.keys()}
        self.best = None

    @staticmethod
    def _get_adjacent_list_item(choice_list, current_value):
        return choice_list[max(0, min(len(choice_list) - 1, choice_list.index(current_value) + np.random.randint(-1, 2)))]

    def modify_hyperparams(self):
        new_hyperparams = copy(self.current)
        parameters = [key for key in self.space.keys() if key not in ('beta_1', 'beta_2', 'eps', 'rho', 'decay')]

        for parameter in parameters:
            new_hyperparams[parameter] = self._get_adjacent_list_item(self.space[parameter], self.current[parameter])

        if new_hyperparams['use_adam']:
            for parameter in ['beta_1', 'beta_2', 'eps']:
                new_hyperparams[parameter] = self._get_adjacent_list_item(self.space[parameter], self.current[parameter])
        else:
            for parameter in ['rho', 'decay']:
                new_hyperparams[parameter] = self._get_adjacent_list_item(self.space[parameter], self.current[parameter])

        self.current = new_hyperparams

    def save_best(self):
        self.best = copy(self.current)


class CrossValidator():
    def __init__(self, n_splits, hyperparams, epochs=4):
        self.n_splits = n_splits
        self.hyperparams = hyperparams
        self.epochs = epochs

    def run(self, images, labels, class_weight):
        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2018).split(images, np.argmax(labels, axis=1)))
        val_scores = []
        for train_idx, validation_idx in folds:
            images_tr = images[train_idx]
            labels_tr = labels[train_idx]
            images_val = images[validation_idx]
            labels_val = labels[validation_idx]

            classifier_trainer = ClassifierTrainer(self.hyperparams)
            classifier_trainer.build_classifier()
            classifier_trainer.compile_classifier()
            classifier_trainer.fit_classifier(images_tr, labels_tr, images_val, labels_val, self.epochs, class_weight)

            val_scores.append(classifier_trainer.classifier.evaluate(images_val, labels_val))

        return val_scores



class ClassifierTrainer:
    def __init__(self, hyperparams, batch_size=32):
        self.hp = hyperparams
        self.batch_size = batch_size
        self.classifier = None

    def build_classifier(self):
        self.classifier = Sequential()
        self.classifier.add(
            Conv2D(filters=self.hp['filters_1'], kernel_size=(self.hp['kernel_1'], self.hp['kernel_1']),
                   padding='Same', activation=self.hp['activation'], input_shape=(56, 56, 1)))
        self.classifier.add(
            Conv2D(filters=self.hp['filters_1'], kernel_size=(self.hp['kernel_1'], self.hp['kernel_1']),
                   padding='Same', activation='relu'))
        self.classifier.add(MaxPool2D(pool_size=(self.hp['pooling_1'], self.hp['pooling_1']), padding='same'))
        if self.hp['use_dropout']:
            self.classifier.add(Dropout(0.25))

        self.classifier.add(
            Conv2D(filters=self.hp['filters_2'], kernel_size=(self.hp['kernel_2'], self.hp['kernel_2']),
                   padding='Same', activation=self.hp['activation']))
        self.classifier.add(
            Conv2D(filters=self.hp['filters_2'], kernel_size=(self.hp['kernel_2'], self.hp['kernel_2']),
                   padding='Same', activation=self.hp['activation']))
        self.classifier.add(MaxPool2D(pool_size=(self.hp['pooling_2'], self.hp['pooling_2']), padding='same'))
        if self.hp['use_dropout']:
            self.classifier.add(Dropout(0.25))

        self.classifier.add(Flatten())
        self.classifier.add(Dense(self.hp['hidden_units'], activation=self.hp['activation']))
        if self.hp['use_dropout']:
            self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(36, activation="softmax"))

    def compile_classifier(self):
        self.classifier.compile(self._optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    @property
    def _optimizer(self):
        if self.hp['use_adam']:
            optimizer = adam(lr=self.hp['learning_rate'], beta_1=self.hp['beta_1'], beta_2=self.hp['beta_2'],
                             epsilon=self.hp['eps'])
        else:
            optimizer = rmsprop(lr=self.hp['learning_rate'], rho=self.hp['rho'], epsilon=self.hp['eps'],
                                decay=self.hp['decay'])
        return optimizer

    @property
    def _callbacks(self):
        callbacks = []
        reduce_lr_callback = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='acc')
        return [reduce_lr_callback] if self.hp['use_lr_reduction'] else None

    def fit_classifier(self, images_tr, labels_tr, images_val=None, labels_val=None, epochs=10, class_weight=None):
        cw = class_weight if self.hp['use_class_weight'] else None

        if self.hp['use_img_augmentation']:
            self._fit(images_tr, labels_tr, images_val, labels_val, epochs, cw)
        else:
            self._fit_generator(images_tr, labels_tr, images_val, labels_val, epochs, cw)

    def _fit(self, images_tr, labels_tr, images_val, labels_val, epochs, class_weight):
        self.classifier.fit(images_tr, labels_tr, batch_size=self.batch_size, epochs=epochs, callbacks=self._callbacks,
                            validation_data=(images_val, labels_val), shuffle=True, class_weight=class_weight)

    def _fit_generator(self, images_tr, labels_tr, images_val, labels_val, epochs, class_weight):
        data_generator = ImageDataGenerator(rotation_range=10, zoom_range=0.10,
                                            width_shift_range=0.10, height_shift_range=0.10)
        data_generator.fit(images_tr)
        self.classifier.fit_generator(data_generator.flow(images_tr, labels_tr, batch_size=self.batch_size),
                                      validation_data=(images_val, labels_val), epochs=epochs, callbacks=self._callbacks,
                                      class_weight=class_weight)



class ClassifierOptimizer():
    def __init__(self, initial_hyperparams=None):
        self.hyperparams_handler = HyperparametersHandler(initial_hyperparams)
        self.best_cv_score = 0
        self.best_hyperparams = None

    def run_optimization(self, rounds, images, labels, class_weight):
        for iteration in range(rounds):
            try:
                cv_score = self._get_cv_score(images, labels, class_weight)

                with open('saved.txt', 'a') as f:
                    log_msg = ' '.join([str(cv_score > self.best_cv_score), str(self.hyperparams_handler.current), 'SCORE:', str(cv_score), '\n\n'])
                    f.write(log_msg)

                if cv_score > self.best_cv_score:
                    self._save_best(cv_score)

            except Exception as e:
                with open('errors.txt', 'a') as f:
                    f.write(str(e) + '\n' + '#' * 50 + '\n')

            self.hyperparams_handler.modify_hyperparams()

    def _get_cv_score(self, images, labels, class_weight):
        cv = CrossValidator(4, self.hyperparams_handler.current)
        scores = cv.run(images, labels, class_weight)
        return np.mean([score[1] for score in scores])


    def _save_best(self, new_best_cv_score):
        self.best_cv_score = new_best_cv_score
        self.hyperparams_handler.save_best()


if __name__ == '__main__':
    images, labels = load_data('train_sample.pkl')
    images = images.reshape(-1, 56, 56, 1)
    class_weight = calculate_class_weight(labels)
    labels = to_categorical(labels, num_classes=36)

    images_tst, labels_tst = load_data('test_sample.pkl')
    images_tst = images_tst.reshape(-1, 56, 56, 1)
    labels_tst = to_categorical(labels_tst, num_classes=36)

    initial_hyperparams = dict()
    initial_hyperparams['filters_1'] = HP_SPACE['filters_1'][1]
    initial_hyperparams['filters_2'] = HP_SPACE['filters_2'][2]
    initial_hyperparams['pooling_1'] = HP_SPACE['pooling_1'][0]
    initial_hyperparams['pooling_2'] = HP_SPACE['pooling_2'][2]
    initial_hyperparams['kernel_1'] = HP_SPACE['kernel_1'][1]
    initial_hyperparams['kernel_2'] = HP_SPACE['kernel_2'][2]
    initial_hyperparams['beta_1'] = HP_SPACE['beta_1'][0]
    initial_hyperparams['beta_2'] = HP_SPACE['beta_2'][1]
    initial_hyperparams['eps'] = HP_SPACE['eps'][1]
    initial_hyperparams['learning_rate'] = HP_SPACE['learning_rate'][1]
    initial_hyperparams['rho'] = HP_SPACE['rho'][0]
    initial_hyperparams['decay'] = HP_SPACE['decay'][2]
    initial_hyperparams['hidden_units'] = HP_SPACE['hidden_units'][2]
    initial_hyperparams['activation'] = HP_SPACE['activation'][1]

    initial_hyperparams['use_adam'] = HP_SPACE['use_adam'][0]
    initial_hyperparams['use_dropout'] = HP_SPACE['use_dropout'][1]
    initial_hyperparams['use_class_weight'] = HP_SPACE['use_class_weight'][0]
    initial_hyperparams['use_lr_reduction'] = HP_SPACE['use_lr_reduction'][0]
    initial_hyperparams['use_img_augmentation'] = HP_SPACE['use_img_augmentation'][1]

    # classifier_optimizer = ClassifierOptimizer(initial_hyperparams)
    # classifier_optimizer.run_optimization(50, images, labels, class_weight)

    classifier_trainer = ClassifierTrainer(initial_hyperparams)
    classifier_trainer.build_classifier()
    classifier_trainer.compile_classifier()
    classifier_trainer.fit_classifier(images, labels, images_tst, labels_tst, epochs=20, class_weight=class_weight)
    # classifier = train_classifier(images, labels, images_tst, labels_tst, initial_hyperparams, class_weight, epochs=10)
    classifier_trainer.classifier.save('keras_cnn_model_2.h5')