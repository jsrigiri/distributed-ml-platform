class OnlineFeatureStore:
    """
    Simple in-memory online feature store.
    Replace later with Redis for production-style serving.
    """
    def __init__(self):
        self.store = {}

    def put_many(self, records, key_col="user_id"):
        for row in records:
            key = row[key_col]
            self.store[key] = row

    def get(self, key):
        return self.store.get(key)

    def exists(self, key):
        return key in self.store