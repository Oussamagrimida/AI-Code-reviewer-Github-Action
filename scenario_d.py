import json

def load_saved_session(session_data: bytes):
    """Restores a user session from previously saved binary data."""
    if not session_data:
        raise ValueError("session_data must not be empty")
    try:
        # Assuming session_data is UTF-8 encoded JSON text
        data_str = session_data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError("session_data is not valid UTF-8") from e
    try:
        return json.loads(data_str)
    except json.JSONDecodeError as e:
        raise ValueError("session_data is not valid JSON") from e