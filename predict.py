import pickle

CURRENT_MODEL = 'sklearn_mlpc_model.pkl'

def load_model():
    with open(CURRENT_MODEL, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(input_data):
    model = load_model()
    return model.predict(input_data)