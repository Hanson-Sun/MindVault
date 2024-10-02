"""
okay so im actually gonna figure this out. Here is how its gonna work.

i have a sqlite database with fts5 search enabled,
- other table to store all the vectors -> index to which file its from the main table

Indexing is simple, just append everything to the sqlite database

Searching is a bit more complicated, but i think i can do it.
- load chunks of the vectors table into memory, and keep the best results in a pq. 
- ideally this should not be done in python, but oh well.
- try to use numpy operations here to speed things up, maybe numba too

Repository layer for database actions, insertions and should be nicely abstracted.

hmm searching might be a bit slower, but this should scale much better.
"""

import sqlite3
from typing import Union
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from nltk.tokenize import sent_tokenize
import os
import json
from sentence_transformers import SentenceTransformer
from torch import cuda
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pathspec
import cProfile
from datetime import datetime
import json
from queue import PriorityQueue
import faiss
import heapq

def initDataBaseConnection(name : str = "test_database.db"):
    conn = sqlite3.connect(name)
    conn.enable_load_extension(True)
    conn.load_extension('./fts5')
    c = conn.cursor()
    c.execute("PRAGMA journal_mode = WAL;")
    c.execute("PRAGMA synchronous = NORMAL;")
    c.execute("PRAGMA temp_store = MEMORY;")
    c.execute("PRAGMA mmap_size = 12000000000;") # 12gb LOL     
    c.execute("PRAGMA cache_size = -500000;")    # 500mb cache  
    # table to store general contents
    c.execute("""
        CREATE TABLE IF NOT EXISTS content_table (
            id INTEGER PRIMARY KEY, 
            file_path TEXT, 
            mod_time REAL, 
            metadata JSON, 
            content TEXT
        )
    """)
    # table to store the fts index, automatically updates when content_table is updated
    c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts_table USING fts5(content, content='content_table', content_rowid='id', tokenize='trigram case_sensitive 1')")
    # table to store the vss index, this needs to be manually updated since it aint support linking (:frown2:)
    c.execute("""
        CREATE TABLE IF NOT EXISTS vss_table(
            id INTEGER PRIMARY KEY,
            content_vector BLOB,
            document_id INTEGER,
            FOREIGN KEY(document_id) REFERENCES content_table(id)
        )
    """)

    # triggers to update the fts_table when content_table is updated
    c.execute("DROP TRIGGER IF EXISTS content_table_ai;")
    c.execute("DROP TRIGGER IF EXISTS content_table_ad;")
    c.execute("DROP TRIGGER IF EXISTS content_table_au;")
    c.execute("""
        CREATE TRIGGER content_table_ai AFTER INSERT ON content_table BEGIN
            INSERT INTO fts_table(rowid, content) VALUES (new.id, new.content);
        END;
    """)
    c.execute("""
        CREATE TRIGGER content_table_ad AFTER DELETE ON content_table BEGIN
            INSERT INTO fts_table(fts_table, rowid, content) VALUES('delete', old.id, old.content);
        END;
    """)
    c.execute("""
        CREATE TRIGGER content_table_au AFTER UPDATE ON content_table BEGIN
            INSERT INTO fts_table(fts_table, rowid, content) VALUES('delete', old.id, old.content);
            INSERT INTO fts_table(rowid, content) VALUES (new.id, new.content);
        END;
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS i2v_index_mapping(
            vector_index_id INTEGER,
            vector_id INTEGER PRIMIARY KEY,
            FOREIGN KEY(vector_id) REFERENCES vss_table(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS vector_indexes(
            index_blob BLOB
        )
    """)
    
    # 0 added, 1 removed, 2 updated
    c.execute("""
        CREATE TABLE IF NOT EXISTS vss_changes_table(
            type INTEGER,
            vector_id INTEGER
        )
    """)
    # triggers to automatically fill the changes table
    c.execute("DROP TRIGGER IF EXISTS vss_table_ai;")
    c.execute("DROP TRIGGER IF EXISTS vss_table_ad;")
    c.execute("DROP TRIGGER IF EXISTS vss_table_au;")
    c.execute("""
        CREATE TRIGGER vss_table_ai AFTER INSERT ON vss_table BEGIN
            INSERT INTO vss_changes_table(type, vector_id) VALUES(0, new.id);
        END;
    """)
    c.execute("""
        CREATE TRIGGER vss_table_ad AFTER DELETE ON vss_table BEGIN
            INSERT INTO vss_changes_table(type, vector_id) VALUES(1, old.id);
        END;
    """)
    c.execute("""
        CREATE TRIGGER vss_table_au AFTER UPDATE ON vss_table BEGIN
            INSERT INTO vss_changes_table(type, vector_id) VALUES(2, old.id);
        END;
    """)

    return conn

# this returns a list of documents
def loadPDF(path : str):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    return pages

def loadText(path : str):
    loader = TextLoader(path)
    data = loader.load()
    return data

def loadMarkdown(path : str):
    loader = UnstructuredMarkdownLoader(path);
    data = loader.load_and_split();
    return data

# returns a single document
def loadCode(path : str):
    load = TextLoader(path)
    data = load.load()
    return data

# figure out how to load images and store the proper metadata
def loadImage(path : str):
    pass

# this processes a list of documents
def chunkifyPDF(data, chunk_sentences = 5):
    for i in range(len(data)):
        sentences = sent_tokenize(data[i].page_content)
        data[i].page_content = [' '.join(sentences[n:n+chunk_sentences]) for n in range(0, len(sentences) + chunk_sentences, chunk_sentences)]
    return data

def chunkifyMarkdown(data, chunk_sentences = 5):
    data = data[0]
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    large_split = markdown_splitter.split_text(data.page_content)
    for i in range(len(large_split)):
        sentences = sent_tokenize(large_split[i].page_content)
        large_split[i].page_content = [' '.join(sentences[n:n+chunk_sentences]) for n in range(0, len(sentences) + chunk_sentences, chunk_sentences)]
        large_split[i].metadata = {**large_split[i].metadata, **data.metadata}

    return large_split

def chunkifyCode(data, language_type:Language, chunk_sentences = 5):
    data = data[0]
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language_type, chunk_size= chunk_sentences * 100, chunk_overlap=0
    )
    large_split = splitter.split_text(data.page_content)
    document = Document(page_content="", metadata={**data.metadata})

    document.page_content = large_split
    return [document]

def getProgrammingLanguage(file_extension : str) -> Union[Language, bool]:
    switcher = {
        ".h" : Language.CPP,
        ".cpp": Language.CPP,
        ".go": Language.GO,
        ".java": Language.JAVA,
        ".kt": Language.KOTLIN,
        ".js": Language.JS,
        ".ts": Language.TS,
        ".php": Language.PHP,
        ".proto": Language.PROTO,
        ".py": Language.PYTHON,
        ".rst": Language.RST,
        ".rb": Language.RUBY,
        ".rs": Language.RUST,
        ".scala": Language.SCALA,
        ".swift": Language.SWIFT,
        ".md": Language.MARKDOWN,
        ".tex": Language.LATEX,
        ".html": Language.HTML,
        ".sol": Language.SOL,
        ".cs": Language.CSHARP,
        ".cbl": Language.COBOL,
        ".c": Language.CPP,
        ".lua": Language.LUA,
        ".pl": Language.PERL,
    }
    return switcher.get(file_extension, False)


def encodeChunks(model, sentences : list):
    return model.encode(sentences)


def indexFile(conn, file_path : str):
    encoded = []
    pages = []
    chunks = []
    filename = os.path.basename(file_path)
    if (file_path.endswith(".pdf")):
        # tqdm.write(f"Indexing PDF: {filename}")
        try:
            pages = loadPDF(file_path)
        except Exception as e:
            print(f"ERROR: {e}")
            return
        chunks = chunkifyPDF(pages)
        for i in tqdm(range(len(chunks)), desc="Encoding chunks"):
            encoded.append(encodeChunks(model, chunks[i].page_content))
    elif (file_path.endswith(".md")):
        # tqdm.write(f"Indexing Markdown: {filename}")
        try:
            pages = loadMarkdown(file_path)
        except Exception as e:
            print(f"ERROR: {e}")
            return
        chunks = chunkifyMarkdown(pages)
        for i in tqdm(range(len(chunks)), desc="Encoding chunks"):
            encoded.append(encodeChunks(model, chunks[i].page_content))
    elif (language := getProgrammingLanguage(os.path.splitext(file_path)[1])):
        # tqdm.write(f"Indexing Code: {filename}")
        try:
            pages = loadCode(file_path)
        except Exception as e:
            print(f"ERROR: {e}")
            return
        chunks = chunkifyCode(pages, language)
        for i in tqdm(range(len(chunks)), desc="Encoding chunks"):
            encoded.append(encodeChunks(model, chunks[i].page_content))
    else:
        tqdm.write("Unsupported file type")
        return
    
    c = conn.cursor()
    
    vss_data = []

    tqdm.write("Inserting into database...")
    c.execute("BEGIN TRANSACTION")
    for i in range(len(pages)):
        c.execute("INSERT INTO content_table (file_path, mod_time, metadata, content) VALUES (?, ?, ?, ?)", 
                  (file_path, os.path.getmtime(file_path), json.dumps(pages[i].metadata), "".join(pages[i].page_content)))
        content_id = c.lastrowid  # Get the id of the last inserted row

        if i >= len(encoded):
            break

        for j in range(len(encoded[i])):
            vss_data.append((content_id, encoded[i][j].tobytes()))

        c.executemany("INSERT INTO vss_table (document_id, content_vector) VALUES (?, ?)", vss_data)
        vss_data.clear()

    try:
        c.execute("COMMIT")
    except sqlite3.OperationalError as e:
        print(f"ERROR: {e}")

    return


def deleteFile(conn, file_path : str):
    c = conn.cursor()
    c.execute("SELECT id FROM content_table WHERE file_path = ?", (file_path,))
    files = c.fetchall()

    c.execute("BEGIN TRANSACTION")
    for page in tqdm(files, desc="Deleting file"):
        # delete row in content table
        c.execute("DELETE FROM content_table WHERE id = ?", (page[0],))
        c.execute("DELETE FROM vss_table WHERE document_id = ?", (page[0],))
    
    try:
        c.execute("COMMIT")
    except sqlite3.OperationalError as e:
        print(f"ERROR: {e}")

    return

def count_files(directory):
    return sum(len(files) for _, _, files in os.walk(directory))

def load_index_ignore(gitignore_path):
    if not os.path.exists(gitignore_path):
        return None
    
    with open(gitignore_path, 'r') as file:
        gitignore = file.read()
    spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, gitignore.splitlines())
    return spec

def is_ignored(path, spec):
    if not spec:
        return False
    return spec.match_file(path)


def indexDirectory(conn, directory_path : str):
    c = conn.cursor()

    total_files = count_files(directory_path)

    ignore_patterns = load_index_ignore(os.path.join(directory_path, ".indexignore"))

    new = []
    changed = []
    deleted = []

    c.execute("""
        CREATE INDEX IF NOT EXISTS index_file_path ON content_table (file_path)
    """)

    tqdm.write("Checking for changes...")
    progress_bar = tqdm(desc="Checking changes", total=total_files)
    for root, _, files in os.walk(directory_path):
        for file in files:
            progress_bar.update()
            file_path = os.path.join(root, file)

            if is_ignored(file_path, ignore_patterns):
                continue

            current_mode_time = os.path.getmtime(file_path)
            c.execute("SELECT mod_time FROM content_table WHERE file_path = ?", (file_path,))
            result = c.fetchone()
            if (result and current_mode_time - result[0] > 0.01):
                changed.append(file_path)
            elif (not result):
                new.append(file_path)
    progress_bar.close()

    c.execute("SELECT file_path FROM content_table")
    file_paths = c.fetchall()
    for file_path in tqdm(file_paths, desc="Checking file existence"):
        file_path = file_path[0]
        if not os.path.exists(file_path):
            deleted.append(file_path)

    c.execute("""
        DROP INDEX IF EXISTS index_file_path
    """)

    tqdm.write("Processing changes...")

    total_operations = len(new) + len(changed) + len(deleted)
    progress_bar = tqdm(total=total_operations, desc="Processing files")

    for file_path in new:
        tqdm.write(f"{file_path} is not indexed. Indexing...")
        indexFile(conn, file_path)
        progress_bar.update()

    for file_path in changed:
        tqdm.write(f"{file_path} has been modified. Reindexing...")
        deleteFile(conn, file_path)
        indexFile(conn, file_path)
        progress_bar.update()

    for file_path in deleted:
        tqdm.write(f"{file_path} does not exist. Deleting...")
        deleteFile(conn, file_path)
        progress_bar.update()

    progress_bar.close()
    return


def generateVectorIndexes(conn, batch_size = 10000):
    tqdm.write("Generating/Updating vector indexes...")

    c = conn.cursor()
    indexes_to_update = set() # list of the ids of any changed indexes
    indexes_to_update_partially = set() # list of the ids of any indexes that need to be updated partially

    # updated vectors 
    c.execute("""
        SELECT * 
        FROM i2v_index_mapping 
        WHERE vector_id IN (
            SELECT vector_id FROM vss_changes_table WHERE type = 2
        )
    """)
    updated = c.fetchall()
    for row in updated:
        indexes_to_update.add(row[0])

    # deleted vectors
    c.execute("""
        SELECT *
        FROM i2v_index_mapping 
        WHERE vector_id IN (
            SELECT vector_id FROM vss_changes_table WHERE type = 1
        )
    """)
    deleted = c.fetchall()
    for row in deleted:
        indexes_to_update.add(row[0])
        c.execute("DELETE FROM i2v_index_mapping WHERE vector_id = ?", (row[1],))

    # added vectors
    c.execute("""
        SELECT * FROM vss_changes_table WHERE type = 0
    """) 
    added = c.fetchall()

    # check the size of the existing index blobs, if they are too small, add the new vectors to them. If they dont fit, create a new index
    c.execute("SELECT rowid FROM vector_indexes")
    vector_indexes = c.fetchall()
    curr_index = 0
    for index in tqdm(vector_indexes, desc="Checking index blobs"):
        c.execute("SELECT COUNT(*) FROM i2v_index_mapping WHERE vector_index_id = ?", (index[0],))
        count = c.fetchone()[0]
        if (curr_index == len(added)):
            break;
        if count < batch_size:
            diff = batch_size - count
            c.executemany("INSERT INTO i2v_index_mapping (vector_index_id, vector_id) VALUES (?, ?)", 
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
        c.execute(query, ids)

        rows = c.fetchall()
        row = [(id, np.frombuffer(vector, dtype='f4')) for id, vector in rows]
        ids = np.array([id for id, _ in row])
        vectors = np.array([])
        if len(row) > 0:
            vectors = np.vstack([vector for _, vector in row])
            index.add_with_ids(vectors, ids)
        else:
            print("No vectors to add to index")
            raise ValueError("No vectors to add to index")

        c.execute("INSERT INTO vector_indexes (index_blob) VALUES (?)", (faiss.serialize_index(index),))
        c.execute("SELECT last_insert_rowid()")
        index_id = c.fetchone()[0]
        c.executemany("INSERT INTO i2v_index_mapping (vector_index_id, vector_id) VALUES (?, ?)", 
                      [(index_id, vector_id[1]) for vector_id in added[i:i+batch_size]])


    for changed_index in tqdm(indexes_to_update, desc="Updating indexes"):
        c.execute("""
            SELECT document_id, content_vector FROM vss_table WHERE id IN (
                SELECT vector_id FROM i2v_index_mapping WHERE vector_index_id = ?
            )
        """, (changed_index,))
        index = faiss.IndexFlatL2(1024)
        index = faiss.IndexIDMap(index)

        rows = c.fetchall()
        row = [(id, np.frombuffer(vector, dtype='f4')) for id, vector in rows]
        ids = np.array([id for id, _ in row])
        vectors = np.array([])
        if len(row) > 0:
            vectors = np.vstack([vector for _, vector in row])
            index.add_with_ids(vectors, ids)
        else:
            print("No vectors to add to index")
            raise ValueError("No vectors to add to index")


        c.execute("UPDATE vector_indexes SET index_blob = ? WHERE rowid = ?", 
                  (faiss.serialize_index(index), changed_index))

    for update_index in tqdm(indexes_to_update_partially, desc="Partially updating indexes"):
        c.execute("SELECT index_blob FROM vector_indexes WHERE rowid = ?", (update_index,))
        result = c.fetchone()
        if not result:
            print("No index found")
            raise ValueError("No index found")
        # print(result)
        index = faiss.deserialize_index(np.frombuffer(result[0], dtype='uint8'))
        # index = faiss.IndexIDMap(index)
        c.execute("""
            SELECT document_id, content_vector FROM vss_table WHERE id IN (
                SELECT vector_id FROM i2v_index_mapping WHERE vector_index_id = ?
            )
        """, (update_index,))
        rows = c.fetchall()
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
        c.execute("UPDATE vector_indexes SET index_blob = ? WHERE rowid = ?", 
                  (faiss.serialize_index(index), update_index))

    # clear everything in the changes table
    c.execute("DELETE FROM vss_changes_table")
    conn.commit()

    return

def fts5Search(conn, query : str, limit : int = 5):
    c = conn.cursor()
    c.execute("SELECT * FROM fts_table WHERE content MATCH ? ORDER BY rank LIMIT ?", (query, limit))
    return c.fetchall()

def incrementalVssSearch(conn, query : str, num_results = 5):
    top_embeddings = []

    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM vector_indexes")
    num_indexes = c.fetchone()[0]
    
    sentences = sent_tokenize(query)
    encoded = encodeChunks(model, sentences)
    # find the average
    query_vector = np.mean(encoded, axis=0).reshape(1, -1)

    for index in tqdm(range(num_indexes), desc="Searching indexes"):
        c.execute("SELECT * FROM vector_indexes WHERE rowid = ?", (index + 1, ))
        result = c.fetchone()
        index = faiss.deserialize_index(np.frombuffer(result[0], dtype='uint8'))

        D, I = index.search(query_vector, num_results)

        for d, i in zip(D[0], I[0]):
            # Invert the distance and add it to the queue
            heapq.heappush(top_embeddings, (-float(d), int(i)))
            # If the queue is too large, remove the smallest item
            if len(top_embeddings) > num_results:
                heapq.heappop(top_embeddings)

        top_embeddings.reverse()

    print(top_embeddings)


    return top_embeddings


def optimize(conn):
    conn.execute("PRAGMA optimize;")

def vacuum(conn):
    conn.execute("PRAGMA vacuum;")

# TODO: please fix this lol
def reduceVssSearchResults(query_vector, results, size : int):
    most_similar_chunks = []

    for result in results:
        sentences = sent_tokenize(result[4])
        max_similarity = -1
        most_similar_chunk = None

        for i in range(len(sentences) - size + 1):
            chunk = sentences[i : i + size]
            if (not chunk): 
                continue
            # chunk = sentences[i : i + size]
            # encoded = encodeChunks(model, chunk)
            # avg_vector = np.mean(encoded, axis=0)
            concatenated_chunk = ' '.join(chunk)
            avg_vector = model.encode(concatenated_chunk)

            # Calculate the cosine similarity
            similarity = cosine_similarity([query_vector], avg_vector)[0][0]
            print(similarity)

            # Save the chunk with the highest similarity
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_chunk = concatenated_chunk

        most_similar_chunks.append({"content": most_similar_chunk, 
                                    "similarity":max_similarity,
                                    "path":result[1],
                                    "metadata":json.loads(result[3])})

    return most_similar_chunks

if __name__ == "__main__":
    model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
    if cuda.is_available():
        model = model.to(cuda.current_device())
    conn = initDataBaseConnection("full_test_new_schema.db")
    indexDirectory(conn, "../../uni-notes")
    generateVectorIndexes(conn)
    c = conn.cursor()
    while True:
        query = input("Enter query: ")
        results = incrementalVssSearch(conn, query)
        results = set(results)
        for result in results:
            print(type(result[1]))
            c.execute("SELECT * FROM content_table WHERE id = ?", (result[1],))
            content = c.fetchone()
            print(f"Similarity {result[0]} at index {result[1]}:")
            print(content)
        # results = incrementalVssSearch(conn, query)
        # print(results)
    conn.close()
