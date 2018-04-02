from sklearn.utils import murmurhash3_32


def hash(token, hash_size):
    return murmurhash3_32(token, positive=True) % hash_size

