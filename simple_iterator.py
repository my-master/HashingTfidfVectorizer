from typing import List

from logger import logger


class SimpleIterator:
    def __init__(self, data, batch_size=None):
        self.data = data
        self.batch_size = batch_size
        self.doc_index = self.get_doc_ids(data)

    def get_doc_ids(self, data: List[str]):
        ids = [i for i in range(len(data))]
        return ids

    def read_batch(self, batch_size=1000):
        _batch_size = self.batch_size or batch_size
        batches = [self.doc_index[i:i + _batch_size] for i in
                   range(0, len(self.doc_index), _batch_size)]
        len_batches = len(batches)

        for i, doc_ids in enumerate(batches):
            logger.info(
                "Processing batch # {} of {} ({} documents)".format(i, len_batches, len(doc_ids)))
            docs = self.data[doc_ids[0]: doc_ids[-1]]
            yield docs, doc_ids
