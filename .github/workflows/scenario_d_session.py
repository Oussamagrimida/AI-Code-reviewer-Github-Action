import pickle

def load_saved_session(session_data: bytes):
    """Restores a user session from previously saved binary data."""
    return pickle.loads(session_data)