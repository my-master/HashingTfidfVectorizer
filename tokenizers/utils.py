from typing import List, Generator, Any


def ngramize(items: List[str], ngram_range=(1, 1)) -> Generator[List[str], Any, None]:
    """
    :param items: list of tokens, lemmas or other strings to form ngrams
    :param ngram_range: range for producing ngrams, ex. for unigrams + bigrams should be set to
    (1, 2), for bigrams only should be set to (2, 2)
    :return: ngrams (as strings) generator
    """

    ngrams = []
    ranges = [(0, i) for i in range(ngram_range[0], ngram_range[1] + 1)]
    for r in ranges:
        ngrams += list(zip(*[items[j:] for j in range(*r)]))

    formatted_ngrams = [' '.join(item) for item in ngrams]

    yield formatted_ngrams
