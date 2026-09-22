def get_order_totals(order_ids, db):
    """Returns a dict of order_id -> total for each order."""
    totals = {}
    if not order_ids:
        return totals
    placeholders = ', '.join('?' for _ in order_ids)
    rows = db.execute(
        f"SELECT id, total FROM orders WHERE id IN ({placeholders})",
        order_ids
    ).fetchall()
    # Map order_id to total from the query results
    totals_from_db = {row[0]: row[1] for row in rows}
    for order_id in order_ids:
        totals[order_id] = totals_from_db.get(order_id)  # Returns None if order_id not found
    return totals