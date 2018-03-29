from typing import List, Tuple, Any, Generator

from logger import logger


class SimpleIterator:
    def __init__(self, data, batch_size=None):
        self.data = data
        self.batch_size = batch_size
        self.doc2index = self.get_doc_ids()

    def get_doc_ids(self):
        return list(range(len(self.data)))

    def read_batch(self, batch_size=1000) -> Generator[Tuple[List[str], List[Any]], Any, None]:
        _batch_size = self.batch_size or batch_size
        batches = [self.doc2index[i:i + _batch_size] for i in
                   range(0, len(self.doc2index), _batch_size)]
        len_batches = len(batches)

        for i, doc_ids in enumerate(batches):
            logger.info(
                "Processing batch # {} of {} ({} documents)".format(i, len_batches, len(doc_ids)))
            docs = [self.data[j] for j in doc_ids]
            yield docs, doc_ids
