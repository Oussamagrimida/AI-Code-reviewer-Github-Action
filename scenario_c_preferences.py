def save_user_preferences(user_id, preferences, db):
    try:
        db.execute(
            "UPDATE users SET preferences = ? WHERE id = ?",
            (preferences, user_id)
        )
        db.commit()
    except Exception:
        pass