import sqlite3
from typing import Union, List
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pathspec
import time
import json
import faiss
import heapq

from encoder import Encoder
from config import SENTENCE_ENCODING_MODEL, INDEX_IGNORE_NAME, CROSS_ENCODER_MODEL, CROSS_ENCODER_TOKENIZER


class VectorDatabase:
    def __init__(self, db_path: str, journal_mode = "WAL", synchronous = "NORMAL", temp_store = "MEMORY", mmap_size = 12000000000, cache_size = -500000):
        self.db_path = db_path

        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()
        self.c.execute(f"PRAGMA journal_mode = {journal_mode};")
        self.c.execute(f"PRAGMA synchronous = {synchronous};")
        self.c.execute(f"PRAGMA temp_store = {temp_store};")
        self.c.execute(f"PRAGMA mmap_size = {mmap_size};")
        self.c.execute(f"PRAGMA cache_size = {cache_size};")

        self.encoder = Encoder(SENTENCE_ENCODING_MODEL)

        self.setup_schema()
    
    def setup_schema(self):
        self.cexecute("""
            CREATE TABLE IF NOT EXISTS content_table (
                id INTEGER PRIMARY KEY, 
                file_path TEXT, 
                mod_time REAL, 
                metadata JSON, 
                content TEXT
            )
        """)
        # table to store the fts index, automatically updates when content_table is updated
        self.cexecute("CREATE VIRTUAL TABLE IF NOT EXISTS fts_table USING fts5(content, content='content_table', content_rowid='id', tokenize='trigram case_sensitive 1')")
        # table to store the vss index, this needs to be manually updated since it aint support linking (:frown2:)
        self.cexecute("""
            CREATE TABLE IF NOT EXISTS vss_table(
                id INTEGER PRIMARY KEY,
                content_vector BLOB,
                document_id INTEGER,
                FOREIGN KEY(document_id) REFERENCES content_table(id)
            )
        """)

        # triggers to update the fts_table when content_table is updated
        self.cexecute("DROP TRIGGER IF EXISTS content_table_ai;")
        self.cexecute("DROP TRIGGER IF EXISTS content_table_ad;")
        self.cexecute("DROP TRIGGER IF EXISTS content_table_au;")
        self.cexecute("""
            CREATE TRIGGER content_table_ai AFTER INSERT ON content_table BEGIN
                INSERT INTO fts_table(rowid, content) VALUES (new.id, new.content);
            END;
        """)
        self.cexecute("""
            CREATE TRIGGER content_table_ad AFTER DELETE ON content_table BEGIN
                INSERT INTO fts_table(fts_table, rowid, content) VALUES('delete', old.id, old.content);
            END;
        """)
        self.cexecute("""
            CREATE TRIGGER content_table_au AFTER UPDATE ON content_table BEGIN
                INSERT INTO fts_table(fts_table, rowid, content) VALUES('delete', old.id, old.content);
                INSERT INTO fts_table(rowid, content) VALUES (new.id, new.content);
            END;
        """)

        self.cexecute("""
            CREATE TABLE IF NOT EXISTS i2v_index_mapping(
                vector_index_id INTEGER,
                vector_id INTEGER PRIMIARY KEY,
                FOREIGN KEY(vector_id) REFERENCES vss_table(id)
            )
        """)

        self.cexecute("""
            CREATE TABLE IF NOT EXISTS vector_indexes(
                index_blob BLOB
            )
        """)
        
        # 0 added, 1 removed, 2 updated
        self.cexecute("""
            CREATE TABLE IF NOT EXISTS vss_changes_table(
                type INTEGER,
                vector_id INTEGER
            )
        """)
        # triggers to automatically fill the changes table
        self.cexecute("DROP TRIGGER IF EXISTS vss_table_ai;")
        self.cexecute("DROP TRIGGER IF EXISTS vss_table_ad;")
        self.cexecute("DROP TRIGGER IF EXISTS vss_table_au;")
        self.cexecute("""
            CREATE TRIGGER vss_table_ai AFTER INSERT ON vss_table BEGIN
                INSERT INTO vss_changes_table(type, vector_id) VALUES(0, new.id);
            END;
        """)
        self.cexecute("""
            CREATE TRIGGER vss_table_ad AFTER DELETE ON vss_table BEGIN
                INSERT INTO vss_changes_table(type, vector_id) VALUES(1, old.id);
            END;
        """)
        self.cexecute("""
            CREATE TRIGGER vss_table_au AFTER UPDATE ON vss_table BEGIN
                INSERT INTO vss_changes_table(type, vector_id) VALUES(2, old.id);
            END;
        """)

    def cexecute(self, sql: str):
        self.c.execute(sql)
        self.conn.commit()
        return self.c.fetchall()

    def reindex_directory(self, directory_path : str):
        if (os.path.isfile(self.db_path)):
            os.remove(self.db_path)
        
        self.index_directory(directory_path)
        return

    def index_directory(self, directory_path : str):

        total_files = self.count_files(directory_path)

        ignore_patterns = self.load_index_ignore(os.path.join(directory_path, INDEX_IGNORE_NAME))

        new = []
        changed = []
        deleted = []

        self.c.execute("""
            CREATE INDEX IF NOT EXISTS index_file_path ON content_table (file_path)
        """)

        tqdm.write("Checking for changes...")
        progress_bar = tqdm(desc="Checking changes", total=total_files)
        for root, _, files in os.walk(directory_path):
            for file in files:
                progress_bar.update()
                file_path = os.path.join(root, file)

                if self.is_ignored(file_path, ignore_patterns):
                    continue

                current_mode_time = os.path.getmtime(file_path)
                self.c.execute("SELECT mod_time FROM content_table WHERE file_path = ?", (file_path,))
                result = self.c.fetchone()
                if (result and current_mode_time - result[0] > 0.01):
                    changed.append(file_path)
                elif (not result):
                    new.append(file_path)
        progress_bar.close()

        self.c.execute("SELECT file_path FROM content_table")
        file_paths = self.c.fetchall()
        for file_path in tqdm(file_paths, desc="Checking file existence"):
            file_path = file_path[0]
            if not os.path.exists(file_path):
                deleted.append(file_path)

        self.c.execute("""
            DROP INDEX IF EXISTS index_file_path
        """)

        tqdm.write("Processing changes...")

        total_operations = len(new) + len(changed) + len(deleted)
        progress_bar = tqdm(total=total_operations, desc="Processing files")

        for file_path in new:
            tqdm.write(f"{file_path} is not indexed. Indexing...")
            self.insert_file(file_path)
            progress_bar.update()

        for file_path in changed:
            tqdm.write(f"{file_path} has been modified. Reindexing...")
            self.delete_file(file_path)
            self.insert_file(file_path)
            progress_bar.update()

        for file_path in deleted:
            tqdm.write(f"{file_path} does not exist. Deleting...")
            self.delete_file(file_path)
            progress_bar.update()

        progress_bar.close()

        tqdm.write("Creating vector indexes...")
        self.generate_vector_indexes()

        return

    def insert_file(self, file_path: str):

        encoded, pages, chunks, success = self.encoder.process_file(file_path)
        if (not success):
            print(f"ERROR: Could not process {file_path}. Insertion failed.")
            return
        
        vss_data = []

        tqdm.write("Inserting into database...")
        self.c.execute("BEGIN TRANSACTION")
        for i in range(len(pages)):
            self.c.execute("INSERT INTO content_table (file_path, mod_time, metadata, content) VALUES (?, ?, ?, ?)", 
                    (file_path, os.path.getmtime(file_path), json.dumps(pages[i].metadata), "".join(pages[i].page_content)))
            content_id = self.c.lastrowid 

            if i >= len(encoded):
                break

            for j in range(len(encoded[i])):
                vss_data.append((content_id, encoded[i][j].tobytes()))

            self.c.executemany("INSERT INTO vss_table (document_id, content_vector) VALUES (?, ?)", vss_data)
            vss_data.clear()

        try:
            self.c.execute("COMMIT")
        except sqlite3.OperationalError as e:
            print(f"ERROR: {e}")

        return
    
    def delete_file(self, file_path: str):
        self.c.execute("SELECT id FROM content_table WHERE file_path = ?", (file_path,))
        files = self.c.fetchall()

        self.c.execute("BEGIN TRANSACTION")
        for page in tqdm(files, desc="Deleting file"):
            # delete row in content table
            self.c.execute("DELETE FROM content_table WHERE id = ?", (page[0],))
            self.c.execute("DELETE FROM vss_table WHERE document_id = ?", (page[0],))
        
        try:
            self.c.execute("COMMIT")
        except sqlite3.OperationalError as e:
            print(f"ERROR: {e}")

        return
    
    def insert_custom(self, encoded, pages, name = "Custom Data"):
        vss_data = []
        tqdm.write("Inserting into database...")
        self.c.execute("BEGIN TRANSACTION")
        for i in range(len(pages)):
            self.c.execute("INSERT INTO content_table (file_path, mod_time, metadata, content) VALUES (?, ?, ?, ?)", 
                    (name, time.time(), json.dumps(pages[i].metadata), "".join(pages[i].page_content)))
            content_id = self.c.lastrowid  # Get the id of the last inserted row

            if i >= len(encoded):
                break

            for j in range(len(encoded[i])):
                vss_data.append((content_id, encoded[i][j].tobytes()))

            self.c.executemany("INSERT INTO vss_table (document_id, content_vector) VALUES (?, ?)", vss_data)
            vss_data.clear()

        try:
            self.c.execute("COMMIT")
        except sqlite3.OperationalError as e:
            print(f"ERROR: {e}")

        return
    
    def load_index_ignore(self, gitignore_path):
        if not os.path.exists(gitignore_path):
            return None
        
        with open(gitignore_path, 'r') as file:
            gitignore = file.read()
        spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, gitignore.splitlines())
        return spec
    
    def is_ignored(self, path, spec):
        if not spec:
            return False
        return spec.match_file(path)
    
    def generate_vector_indexes(self, batch_size = 10000):
        tqdm.write("Generating/Updating vector indexes...")

        indexes_to_update = set() # list of the ids of any changed indexes
        indexes_to_update_partially = set() # list of the ids of any indexes that need to be updated partially

        # updated vectors 
        self.c.execute("""
            SELECT * 
            FROM i2v_index_mapping 
            WHERE vector_id IN (
                SELECT vector_id FROM vss_changes_table WHERE type = 2
            )
        """)
        updated = self.c.fetchall()
        for row in updated:
            indexes_to_update.add(row[0])

        # deleted vectors
        self.c.execute("""
            SELECT *
            FROM i2v_index_mapping 
            WHERE vector_id IN (
                SELECT vector_id FROM vss_changes_table WHERE type = 1
            )
        """)
        deleted = self.c.fetchall()
        for row in deleted:
            indexes_to_update.add(row[0])
            self.c.execute("DELETE FROM i2v_index_mapping WHERE vector_id = ?", (row[1],))

        # added vectors
        self.c.execute("""
            SELECT * FROM vss_changes_table WHERE type = 0
        """) 
        added = self.c.fetchall()

        # check the size of the existing index blobs, if they are too small, add the new vectors to them. If they dont fit, create a new index
        self.c.execute("SELECT rowid FROM vector_indexes")
        vector_indexes = self.c.fetchall()
        curr_index = 0
        for index in tqdm(vector_indexes, desc="Checking index blobs"):
            self.c.execute("SELECT COUNT(*) FROM i2v_index_mapping WHERE vector_index_id = ?", (index[0],))
            count = self.c.fetchone()[0]
            if (curr_index == len(added)):
                break;
            if count < batch_size:
                diff = batch_size - count
                self.c.executemany("INSERT INTO i2v_index_mapping (vector_index_id, vector_id) VALUES (?, ?)", 
                            [(index[0], row[1]) for row in added[curr_index:curr_index + diff]])
                indexes_to_update_partially.add(index[0])
                curr_index += diff
                break

        # if there are still vectors left, create a new index
        # only insert up to batch size tho
        for i in tqdm(range(curr_index, len(added), batch_size), desc="Creating new indexes"):
            index = faiss.IndexFlatL2(1024)
            index = faiss.IndexIDMap(index)

            ids = [row[1] for row in added[i:i+batch_size]]
            placeholders = ', '.join('?' for _ in ids)
            query = f"SELECT document_id, content_vector FROM vss_table WHERE id IN ({placeholders})"
            self.c.execute(query, ids)

            rows = self.c.fetchall()
            row = [(id, np.frombuffer(vector, dtype='f4')) for id, vector in rows]
            ids = np.array([id for id, _ in row])
            vectors = np.array([])
            if len(row) > 0:
                vectors = np.vstack([vector for _, vector in row])
                index.add_with_ids(vectors, ids)
            else:
                print("No vectors to add to index")
                raise ValueError("No vectors to add to index")

            self.c.execute("INSERT INTO vector_indexes (index_blob) VALUES (?)", (faiss.serialize_index(index),))
            self.c.execute("SELECT last_insert_rowid()")
            index_id = self.c.fetchone()[0]
            self.c.executemany("INSERT INTO i2v_index_mapping (vector_index_id, vector_id) VALUES (?, ?)", 
                        [(index_id, vector_id[1]) for vector_id in added[i:i+batch_size]])


        for changed_index in tqdm(indexes_to_update, desc="Updating indexes"):
            self.c.execute("""
                SELECT document_id, content_vector FROM vss_table WHERE id IN (
                    SELECT vector_id FROM i2v_index_mapping WHERE vector_index_id = ?
                )
            """, (changed_index,))
            index = faiss.IndexFlatL2(1024)
            index = faiss.IndexIDMap(index)

            rows = self.c.fetchall()
            row = [(id, np.frombuffer(vector, dtype='f4')) for id, vector in rows]
            ids = np.array([id for id, _ in row])
            vectors = np.array([])
            if len(row) > 0:
                vectors = np.vstack([vector for _, vector in row])
                index.add_with_ids(vectors, ids)
            else:
                print("No vectors to add to index")
                raise ValueError("No vectors to add to index")


            self.c.execute("UPDATE vector_indexes SET index_blob = ? WHERE rowid = ?", 
                    (faiss.serialize_index(index), changed_index))

        for update_index in tqdm(indexes_to_update_partially, desc="Partially updating indexes"):
            self.c.execute("SELECT index_blob FROM vector_indexes WHERE rowid = ?", (update_index,))
            result = self.c.fetchone()
            if not result:
                print("No index found")
                raise ValueError("No index found")
            # print(result)
            index = faiss.deserialize_index(np.frombuffer(result[0], dtype='uint8'))
            # index = faiss.IndexIDMap(index)
            self.c.execute("""
                SELECT document_id, content_vector FROM vss_table WHERE id IN (
                    SELECT vector_id FROM i2v_index_mapping WHERE vector_index_id = ?
                )
            """, (update_index,))
            rows = self.c.fetchall()
            row = [(id, np.frombuffer(vector, dtype='f4')) for id, vector in rows]
            ids = np.array([id for id, _ in row])
            vectors = np.array([])
            if len(row) > 0:
                vectors = np.vstack([vector for _, vector in row])
                index.add_with_ids(vectors, ids)
            else:
                print("No vectors to add to index")
                raise ValueError("No vectors to add to index")
            
            index.add_with_ids(vectors, ids)
            self.c.execute("UPDATE vector_indexes SET index_blob = ? WHERE rowid = ?", 
                    (faiss.serialize_index(index), update_index))

        # clear everything in the changes table
        self.c.execute("DELETE FROM vss_changes_table")
        self.conn.commit()

        return
    
    def fts5_search(self, query : str, limit : int = 5):
        # self.c.execute("SELECT * FROM fts_table WHERE content MATCH ? ORDER BY rank LIMIT ?", (query, limit))
        self.c.execute("""
            SELECT content_table.file_path, content_table.metadata, content_table.content
            FROM content_table
            JOIN fts_table ON content_table.id = fts_table.rowid
            WHERE fts_table MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))
        return self.c.fetchall()

    def vss_search(self, query : str, num_results = 5, return_content = True, return_ranking = False):
        top_embeddings = []

        self.c.execute("SELECT COUNT(*) FROM vector_indexes")
        num_indexes = self.c.fetchone()[0]
        
        sentences = self.encoder.tokenize(query)
        encoded = self.encoder.encode_chunks(sentences)

        # find the average
        query_vector = np.mean(encoded, axis=0).reshape(1, -1)

        for index in tqdm(range(num_indexes), desc="Searching indexes"):
            self.c.execute("SELECT * FROM vector_indexes WHERE rowid = ?", (index + 1, ))
            result = self.c.fetchone()
            index = faiss.deserialize_index(np.frombuffer(result[0], dtype='uint8'))

            D, I = index.search(query_vector, num_results)

            for d, i in zip(D[0], I[0]):
                # Invert the distance and add it to the queue
                heapq.heappush(top_embeddings, (-float(d), int(i)))
                # If the queue is too large, remove the smallest item
                if len(top_embeddings) > num_results:
                    heapq.heappop(top_embeddings)

            top_embeddings.reverse()

        if (return_content):
            content = []
            self.c.execute("BEGIN TRANSACTION")
            for (distance, id) in top_embeddings:
                self.c.execute("SELECT * FROM content_table WHERE id = ?", (id,))
                content.append(self.c.fetchone())
            self.c.execute("COMMIT")

            if (not return_ranking):
                for i in range(len(top_embeddings)):
                    top_embeddings[i] = content[i]
            else:
                for i in range(len(top_embeddings)):
                    top_embeddings[i] = (top_embeddings[i][0], content[i])
        else:
            if (not return_ranking):
                for i in range(len(top_embeddings)):
                    top_embeddings[i] = top_embeddings[i][1]

        return top_embeddings
    
    def cross_encoder_reranking(self, query: str, results : List[str]) -> List[str]:
        # create query and results tuples
        query_result_pairs = []
        max_length = 512
        # for result in results:
        #     # print(result[2])
        #     # result_content = str(result[2]).strip() if result[2] is not None else ""
        #     query_result_pairs.append((str(query), str(result[2])))

        # okay so there has to be a better way to this this no? cuz i want to do a weighted ranking system
        # split larger results into smaller chunks and sort relevance based on that, using those values to then 
        # rerank the final query...
        for result in results:
            tokens_query = CROSS_ENCODER_TOKENIZER.tokenize(str(query))
            tokens_result = CROSS_ENCODER_TOKENIZER.tokenize(str(result[2]))
            num_tokens_available = max_length - 3  # 3 tokens for [CLS], [SEP], [SEP]
            num_tokens_query = len(tokens_query)
            num_tokens_result = len(tokens_result)
            half_available = num_tokens_available // 2

            # Truncate tokens if their combined length exceeds the maximum
            if num_tokens_query + num_tokens_result > num_tokens_available:
                tokens_query = tokens_query[:min(half_available, num_tokens_query)]
                tokens_result = tokens_result[:min(half_available, num_tokens_result)]
            # elif num_tokens_query + num_tokens_result < num_tokens_available:
            #     tokens_result += [CROSS_ENCODER_TOKENIZER.pad_token] * max(half_available)
            #     tokens_query += [CROSS_ENCODER_TOKENIZER.pad_token] * max(half_available)
            
            # if num_tokens_query + num_tokens_result > num_tokens_available:


            truncated_query = CROSS_ENCODER_TOKENIZER.convert_tokens_to_string(tokens_query)
            truncated_result = CROSS_ENCODER_TOKENIZER.convert_tokens_to_string(tokens_result)
            query_result_pairs.append((truncated_query, truncated_result))

        scores = CROSS_ENCODER_MODEL.predict(query_result_pairs)

        print(scores)
        # ok so i need to sort results based on the scores
        sorted_results = [x for _, x in sorted(zip(scores, results), key=lambda pair: pair[0], reverse=True)]

        return sorted_results
    
    def hybrid_search(self, query: str, query_num: int = 5) -> List[str]:
        # perform vector search first, then use fts5 search, and combine the results
        vss_results = self.vss_search(query, query_num, return_ranking = False)
        fts_results = self.fts5_search(query, query_num)

        # print(vss_results)
        # print(fts_results)

        joined_results = vss_results + fts_results

        reranked_results = self.cross_encoder_reranking(query, joined_results)
        return reranked_results

    def optimize(self):
        self.conn.execute("PRAGMA optimize;")

    def vacuum(self):
        self.conn.execute("PRAGMA vacuum;")

    @staticmethod
    def count_files(directory):
        return sum(len(files) for _, _, files in os.walk(directory))
