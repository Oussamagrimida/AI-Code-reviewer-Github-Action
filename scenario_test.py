import pickle

def load_session(data: bytes):
    return pickle.loads(data)
