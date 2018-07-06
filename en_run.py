import time

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from iterators.sqlite_iterator import SQLiteDataIterator
from iterators.simple_iterator import SimpleIterator

from tokenizers.simple_tokenizer import SimpleTokenizer
from tokenizers.spacy_tokenizer import SpacyTokenizer

from vectorizer import HashingTfIdfVectorizer
from logger import logger

DATA_PATH = '/media/olga/Data/projects/ODQA/data/wiki_db/test.db'
SAVE_PATH = '/media/olga/Data/projects/ODQA/data/wiki_db/test'
LOAD_PATH = '/media/olga/Data/projects/ODQA/data/wiki_db/test.npz'

data = ["I think it's better to fry mushrooms.",
        "Oh, this senseless life of ours!"] * 20000
# data = ['Hello world', 'Shiny day']

# iterator = SQLiteDataIterator(DATA_PATH, batch_size=1000)
iterator = SimpleIterator(data, batch_size=1000)
vectorizer = HashingTfIdfVectorizer(iterator, tokenizer=SimpleTokenizer(ngram_range=(1, 2),
                                                                        stopwords=ENGLISH_STOP_WORDS))

start_time = time.time()

try:
    vectorizer.fit()
    vectorizer.save(SAVE_PATH)
    vectorizer.load(LOAD_PATH)
    vectorizer.transform(data)
except Exception as e:
    logger.exception(e)
    raise

tfidf_matrix = vectorizer.load('/media/olga/Data/projects/ODQA/data/wiki_db/test.npz')

logger.info("Completed in {} s.".format(time.time() - start_time))
