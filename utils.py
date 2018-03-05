from typing import Tuple, Dict

from sklearn.utils import murmurhash3_32
import scipy as sp
from scipy import sparse
import numpy as np


def hash(token, hash_size):
    return murmurhash3_32(token, positive=True) % hash_size


def load_tfidf_matrix(load_path) -> Tuple[sp.sparse.csr_matrix, Dict]:
    loader = np.load(load_path)
    matrix = sp.sparse.csr_matrix((loader['data'], loader['indices'],
                                   loader['indptr']), shape=loader['shape'])
    return matrix, loader['opts'].item(0)
