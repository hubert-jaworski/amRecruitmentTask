import pickle
import keras
import numpy as np

from keras.utils import to_categorical

CURRENT_MODEL = 'keras_cnn_model_2.h5'

def load_model():
    if CURRENT_MODEL.endswith('.pkl'):
        with open(CURRENT_MODEL, 'rb') as f:
            model = pickle.load(f)
    else:
        model = keras.models.load_model(CURRENT_MODEL)
    return model

def predict(input_data):
    model = load_model()
    return np.argmax(model.predict(input_data), axis=1)




if __name__ == '__main__':
    from utils import load_data
    images_tst, labels_tst = load_data('test_sample.pkl')
    images_tst = images_tst.reshape(-1, 56, 56, 1)
    labels_tst = to_categorical(labels_tst, num_classes=36)

    model = load_model()

    print(model.evaluate(images_tst, labels_tst))
