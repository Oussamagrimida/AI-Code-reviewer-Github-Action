import logging
import json
def save_user_preferences(user_id, preferences, db):
    if user_id is None:
        raise ValueError("user_id must not be None")
    try:
        db.execute(
            "UPDATE users SET preferences = ? WHERE id = ?",
            (json.dumps(preferences), user_id)
        )
        db.commit()
    except Exception as e:
        logging.error(f"Error saving user preferences for user {user_id}: {e}")
        raise