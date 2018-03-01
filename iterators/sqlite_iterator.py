import sqlite3

from logger import logger


class SQLiteDataIterator:
    def __init__(self, load_path: str, batch_size=None):
        self.load_path = load_path
        self.connect = sqlite3.connect(self.load_path, check_same_thread=False)
        self.db_name = self.get_db_name()
        self.doc_ids = self.get_doc_ids()
        self.doc_index = self.map_doc2idx()
        self.batch_size = batch_size

    def get_doc_ids(self):
        cursor = self.connect.cursor()
        cursor.execute('SELECT id FROM {}'.format(self.db_name))
        ids = [ids[0] for ids in cursor.fetchall()]
        cursor.close()
        return ids

    def get_db_name(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        assert cursor.arraysize == 1
        name = cursor.fetchmany(0)[0][0]
        cursor.close()
        return name

    def map_doc2idx(self):
        doc2idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        print("The size of database is {} documents".format(len(doc2idx)))
        return doc2idx

    def get_doc_content(self, doc_id):
        cursor = self.connect.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (doc_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    def read_batch(self, batch_size=1000):
        _batch_size = self.batch_size or batch_size
        batches = [self.doc_ids[i:i + _batch_size] for i in
                   range(0, len(self.doc_ids), _batch_size)]
        len_batches = len(batches)

        for i, doc_ids in enumerate(batches):
            logger.info(
                "Processing batch # {} of {} ({} documents)".format(i, len_batches, len(doc_ids)))
            docs = [self.get_doc_content(doc_id) for doc_id in doc_ids]
            yield docs, doc_ids
