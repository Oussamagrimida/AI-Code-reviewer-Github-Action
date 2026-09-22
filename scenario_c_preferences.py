import logging
def save_user_preferences(user_id, preferences, db):
    try:
        db.execute(
            "UPDATE users SET preferences = ? WHERE id = ?",
            (preferences, user_id)
        )
        db.commit()
    except Exception as e:
        logging.error(f"Error saving user preferences for user {user_id}: {e}")
        raise