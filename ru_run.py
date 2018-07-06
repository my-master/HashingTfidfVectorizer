import time

from nltk.corpus import stopwords

STOPWORDS = stopwords.words('russian')

from iterators.sqlite_iterator import SQLiteDataIterator
from tokenizers.ru_tokenizer import RussianTokenizer

from vectorizer import HashingTfIdfVectorizer
from logger import logger

DATA_PATH = '/media/olga/Data/projects/ODQA/data/wiki_db/ruwiki.db'
SAVE_PATH = '/media/olga/Data/projects/ODQA/data/wiki_db/ruwiki_tfidf_matrix.npz'

iterator = SQLiteDataIterator(DATA_PATH, batch_size=1000)
vectorizer = HashingTfIdfVectorizer(iterator,
                                    tokenizer=RussianTokenizer(ngram_range=(1, 2), lemmas=True,
                                                               stopwords=STOPWORDS))

start_time = time.time()

try:
    vectorizer.fit()
    vectorizer.save(SAVE_PATH)
    logger.info("Completed successfully in {} s.".format(time.time() - start_time))
except Exception as e:
    logger.info("Completed with exceptions in {} s.".format(time.time() - start_time))
    logger.exception(e)
    raise
