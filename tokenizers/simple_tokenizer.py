from typing import List, Generator, Any

from logger import logger


class SimpleTokenizer:
    """
    Tokenize or lemmatize a list of documents.
    Return list of tokens or lemmas, without sentencizing.
    Works only for English language.
    """

    def __init__(self, stopwords=None, batch_size=None, ngram_range=None):
        """
        :param disable: pipeline processors to omit; if nothing should be disabled,
         pass an empty list
        :param stopwords: a set of words to skip
        """

        self.stopwords = stopwords or []
        self.batch_size = batch_size
        self.ngram_range = ngram_range

    def lemmatize(self, data: List[str], ngram_range=(1, 1), batch_size=10000, n_threads=1) -> \
            Generator[List[str], Any, None]:
        """
        Lemmatize a list of documents.
        :param data: a list of documents to process
        :param ngram_range: range for producing ngrams, ex. for unigrams + bigrams should be set to
        (1, 2), for bigrams only should be set to (2, 2)
        :param batch_size: the number of documents to process at once;
        improves the spacy 'pipe' performance; shouldn't be too small
        :param n_threads: a number of threads for parallel computing; doesn't work good
         on a standard Python
        :return: a single processed doc generator
        """
        size = len(data)

        _ngram_range = self.ngram_range or ngram_range

        for i, doc in enumerate(data):
            logger.debug("Lemmatize doc {} from {}".format(i, size))
            lemmas = doc.split()
            processed_doc = self.ngramize(lemmas, ngram_range=_ngram_range)
            yield from processed_doc

    def ngramize(self, items: List[str], ngram_range=(1, 1)) -> Generator[List[str], Any, None]:
        """
        :param items: list of tokens, lemmas or other strings to form ngrams
        :param ngram_range: range for producing ngrams, ex. for unigrams + bigrams should be set to
        (1, 2), for bigrams only should be set to (2, 2)
        :return:
        """
        _ngram_range = self.ngram_range or ngram_range

        filtered = list(
            filter(lambda x: x.isalpha() and x not in self.stopwords, items))

        ngrams = []
        ranges = [(0, i) for i in range(_ngram_range[0], _ngram_range[1] + 1)]
        for r in ranges:
            ngrams += list(zip(*[filtered[j:] for j in range(*r)]))

        formatted_ngrams = [' '.join(item) for item in ngrams]

        yield formatted_ngrams

    def set_stopwords(self, stopwords):
        self.stopwords = stopwords