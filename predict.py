import pickle
import keras
import numpy as np

CURRENT_MODEL = 'keras_cnn_model.h5'

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