from typing import List, Generator, Any, Tuple

from logger import logger
from tokenizers.utils import ngramize


class SimpleTokenizer:
    """
    Tokenize a list of documents by simple split().
    Return list of tokens, without sentencizing.
    """

    def __init__(self, stopwords=None, ngram_range: Tuple[int, int] = None,
                 lowercase: bool = None, alphas_only: bool = None):
        """
        :param stopwords: a set of words to skip
        :param ngram_range: range for producing ngrams, ex. for unigrams + bigrams should be set to
        (1, 2), for bigrams only should be set to (2, 2)
        :param lowercase: whether to perform lowercasing or not
        :param alphas_only: should filter numeric and alpha-numeric types or not
        """
        self._stopwords = stopwords or []
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        self.alphas_only = alphas_only

    @property
    def stopwords(self):
        return self._stopwords

    @stopwords.setter
    def stopwords(self, stopwords: List[str]):
        self._stopwords = stopwords

    def tokenize(self, data: List[str], ngram_range=(1, 1), lowercase=True) -> \
            Generator[List[str], Any, None]:
        """
        Tokenize a list of documents.
        :param data: a list of documents to process
        :param ngram_range: range for producing ngrams, ex. for unigrams + bigrams should be set to
        (1, 2), for bigrams only should be set to (2, 2)
        :param lowercase: whether to perform lowercasing or not
        :return: a single processed doc generator
        """
        size = len(data)

        _ngram_range = self.ngram_range or ngram_range
        _lowercase = self.lowercase or lowercase

        for i, doc in enumerate(data):
            logger.debug("Tokenize doc {} from {}".format(i, size))
            if _lowercase:
                tokens = doc.lower().split()
            else:
                tokens = doc.split()
            filtered = self._filter(tokens)
            processed_doc = ngramize(filtered, ngram_range=_ngram_range)
            yield from processed_doc

    def sklearn_tokenize(self, data: str):
        """
        Use as argument for `tokenizer` kw in sklearn TfidfVectorizer
        """
        return list(*self.tokenize([data]))

    def _filter(self, items, alphas_only=True):
        """
        Make ngrams from a list of tokens/lemmas
        :param items: list of tokens, lemmas or other strings to form ngrams
        :param alphas_only: should filter numeric and alpha-numeric types or not
        :return: filtered list of tokens/lemmas
        """
        _alphas_only = self.alphas_only or alphas_only

        if _alphas_only:
            filter_fn = lambda x: x.isalpha() and x not in self._stopwords
        else:
            filter_fn = lambda x: x not in self._stopwords

        return list(filter(filter_fn, items))

