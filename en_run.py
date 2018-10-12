import time

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from iterators.sqlite_iterator import SQLiteDataIterator
# from iterators.simple_iterator import SimpleIterator

# from tokenizers.simple_tokenizer import SimpleTokenizer
from tokenizers.stream_spacy_tokenizer import StreamSpacyTokenizer

from vectorizer import HashingTfIdfVectorizer
from logger import logger

DATA_PATH = '/media/olga/Data/projects/DeepPavlov/download/odqa/enwiki_test.db'
SAVE_PATH = '/media/olga/Data/projects/DeepPavlov/download/odqa/enwiki_test_1.npz'

# data = ["I think it's better to fry mushrooms.",
#         "Oh, this senseless life of ours!"] * 20000
data = ['Hello world', 'Shiny day']

iterator = SQLiteDataIterator(DATA_PATH, batch_size=1000)
# iterator = SimpleIterator(data, batch_size=1000)
# vectorizer = HashingTfIdfVectorizer(iterator, tokenizer=SimpleTokenizer(ngram_range=(1, 2),
#                                                                         stopwords='sklearn'))
vectorizer = HashingTfIdfVectorizer(iterator, tokenizer=StreamSpacyTokenizer(ngram_range=[1, 2],
                                                                             lemmas=True,
                                                                             stopwords='sklearn',
                                                                             n_threads=4))
start_time = time.time()

try:
    vectorizer.fit()
    vectorizer.save(SAVE_PATH)
    # DEBUG
    # vectorizer.load(SAVE_PATH)
    # vectorizer.transform(data)
except Exception as e:
    logger.exception(e)
    raise

# DEBUG
# vectorizer.load(SAVE_PATH)
# logger.info("Completed in {} s.".format(time.time() - start_time))
