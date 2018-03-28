from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import concurrent
from typing import TypeVar, List, Any, Generator, Tuple, KeysView, ValuesView, Dict

import scipy as sp
from scipy import sparse
import numpy as np

from tokenizers.spacy_tokenizer import SpacyTokenizer
from logger import logger
from utils import hash

T = TypeVar('T')
TOKENIZER = None


class HashingTfIdfVectorizer:
    """
    Create a tfidf matrix from collection of documents.
    """

    def __init__(self, data_iterator: T, hash_size=2 ** 24, ngram_range=(1, 1),
                 stopwords=None, tokenizer=SpacyTokenizer(), batch_size=None):
        """

        :param data_iterator: an instance of an iterator class, producing data batches;
        the iterator class should implement read_batch() method
        :param hash_size: a size of hash, power of 2
        :param tokenizer: an instance of a tokenizer class
        """
        self.doc2index = data_iterator.doc2index
        self.hash_size = hash_size
        self.text_processor = tokenizer.lemmatize
        self.data_iterator = data_iterator
        self.ngram_range = ngram_range
        self.freqs = None

        if stopwords:
            tokenizer.stopwords = stopwords

        if batch_size:
            tokenizer.batch_size = batch_size

        global TOKENIZER
        TOKENIZER = tokenizer

    def get_counts(self, docs: List[str], doc_ids: List[Any]) \
            -> Generator[Tuple[KeysView, ValuesView, List[int]], Any, None]:
        logger.info("Tokenizing batch...")
        batch_ngrams = list(self.text_processor(docs, ngram_range=self.ngram_range))
        logger.info("Counting hash...")
        doc_id = iter(doc_ids)
        for ngrams in batch_ngrams:
            counts = Counter([hash(gram, self.hash_size) for gram in ngrams])
            hashes = counts.keys()
            values = counts.values()
            _id = self.doc2index[next(doc_id)]
            if values:
                col_id = [_id] * len(values)
            else:
                col_id = []
            yield hashes, values, col_id

    @staticmethod
    def get_counts_parallel(kwargs) -> Tuple[List[int], List[int], List[int]]:
        """
        Get batch counts. The same as get_counts(), but rewritten as staticmethod to be suitable
        for parallelization.
        """

        docs = kwargs['docs']
        doc_ids = kwargs['doc_ids']
        index = kwargs['doc2index']
        hash_size = kwargs['hash_size']
        ngram_range = kwargs['ngram_range']

        logger.info("Tokenizing batch...")
        batch_ngrams = list(TOKENIZER.lemmatize(docs, ngram_range=ngram_range))
        doc_id = iter(doc_ids)

        batch_hashes = []
        batch_values = []
        batch_col_ids = []

        logger.info("Counting hash...")
        for ngrams in batch_ngrams:
            counts = Counter([hash(gram, hash_size) for gram in ngrams])
            hashes = counts.keys()
            values = counts.values()
            col_id = [index[next(doc_id)]] * len(values)
            batch_hashes.extend(hashes)
            batch_values.extend(values)
            batch_col_ids.extend(col_id)

        return batch_hashes, batch_values, batch_col_ids

    def get_count_matrix(self, row: List[int], col: List[int], data: List[int], size) \
            -> sp.sparse.csr_matrix:
        count_matrix = sparse.csr_matrix((data, (row, col)), shape=(self.hash_size, size))
        count_matrix.sum_duplicates()
        return count_matrix

    @staticmethod
    def get_tfidf_matrix(count_matrix: sp.sparse.csr_matrix) -> Tuple[sp.sparse.csr_matrix, np.array]:
        """Convert a word count matrix into a tfidf matrix."""

        binary = (count_matrix > 0).astype(int)
        term_freqs = np.array(binary.sum(1)).squeeze()
        idfs = np.log((count_matrix.shape[1] - term_freqs + 0.5) / (term_freqs + 0.5))
        idfs[idfs < 0] = 0
        idfs = sp.sparse.diags(idfs, 0)
        tfs = count_matrix.log1p()
        tfidfs = idfs.dot(tfs)
        return tfidfs, term_freqs

    def fit(self) -> sp.sparse.csr_matrix:
        rows = []
        cols = []
        data = []

        for docs, doc_ids in self.data_iterator.read_batch():
            for batch_rows, batch_data, batch_cols in self.get_counts(docs, doc_ids):
                rows.extend(batch_rows)
                cols.extend(batch_cols)
                data.extend(batch_data)

        count_matrix = self.get_count_matrix(rows, cols, data, size=len(self.doc2index))
        tfidf_matrix, term_freqs = self.get_tfidf_matrix(count_matrix)
        self.freqs = term_freqs
        return tfidf_matrix

    def fit_parallel(self, n_jobs=1) -> sp.sparse.csr_matrix:

        rows = []
        cols = []
        data = []

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(self.get_counts_parallel, (
                {'docs': docs, 'doc_ids': doc_ids, 'doc2index': self.doc2index,
                 'hash_size': self.hash_size, 'ngram_range': self.ngram_range})) for docs, doc_ids
                       in
                       self.data_iterator.read_batch()]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                rows.extend(result[0])
                data.extend(result[1])
                cols.extend(result[2])

        count_matrix = self.get_count_matrix(rows, cols, data, size=len(self.doc2index))
        tfidf_matrix, term_freqs = self.get_tfidf_matrix(count_matrix)
        self.freqs = term_freqs
        return tfidf_matrix

    def save_matrix(self, save_path, tfidf_matrix: sp.sparse.csr_matrix) -> None:

        opts = {'hash_size': self.hash_size,
                'ngram_rage': self.ngram_range,
                'doc2index': self.doc2index,
                'term_freqs': self.freqs}

        data = {
            'data': tfidf_matrix.data,
            'indices': tfidf_matrix.indices,
            'indptr': tfidf_matrix.indptr,
            'shape': tfidf_matrix.shape,
            'opts': opts
        }
        np.savez(save_path, **data)
