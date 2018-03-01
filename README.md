# HashingTfidfVectorizer
Very fast implementation of tf-idf vectorizer.

## Features

* data batch iteration
* hash
* parallel computing
* fast implementation of [SpaCy](https://spacy.io/) tokenizer
* SQLite iterator (it's not necessary to use it, but if you have a SQLite
textual database, it may be fun)

Though I'm still working on imporving of the parallel computing part.

## Usage

```python
import time

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from tokenizers.simple_iterator import SimpleIterator
from vectorizer import HashingTfIdfVectorizer

DATA = ["I think it's better to fry mushrooms.",
        "Oh, this senseless life of ours!"] * 20000

iterator = SimpleIterator(DATA, batch_size=1000)
vectorizer = HashingTfIdfVectorizer(iterator, ngram_range=(1, 2), stopwords=ENGLISH_STOP_WORDS,
                                    batch_size=1000, tokenizer=SpacyTokenizer())

t01 = time.time()
tfidf_matrix = vectorizer.fit_parallel(n_jobs=7)
t1 = time.time() - t01

t02 = time.time()
tfidf_matrix = vectorizer.fit()
t2 = time.time() - t02


print(
    'Process time for parallel fit, {} docs: {} s.'.format(len(iterator.doc_index), t1))

print(
    'Process time for non parallel fit, {} docs: {} s.'.format(len(iterator.doc_index), t2))
```


```
Process time for parallel fit, 40000 docs: 9.25651478767395 s.
Process time for non parallel fit, 40000 docs: 12.76369833946228 s.
```

