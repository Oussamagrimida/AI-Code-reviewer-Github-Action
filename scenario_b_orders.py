def get_order_totals(order_ids, db):
    """Returns a dict of order_id -> total for each order."""
    totals = {}
    for order_id in order_ids:
        row = db.execute("SELECT total FROM orders WHERE id = ?", (order_id,))
        totals[order_id] = row.fetchone()[0]
    return totals