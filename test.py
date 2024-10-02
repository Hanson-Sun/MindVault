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
import re
from nltk.tokenize import sent_tokenize
import os
import json
from sentence_transformers import SentenceTransformer
from torch import cuda
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pathspec
import pygit2
from git import Repo
import cProfile
from datetime import datetime
import hashlib
import json

def initDataBaseConnection(name : str = "test_database.db"):
    conn = sqlite3.connect(name)
    conn.enable_load_extension(True)
    conn.load_extension('./vector0')
    conn.load_extension('./vss0')
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
    c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vss_table USING vss0(content_vector(1024))")
    # table to store the mapping between the content_table and the vss_table
    c.execute("""
        CREATE TABLE IF NOT EXISTS id_mapping (
            content_id INTEGER,
            vss_rowid INTEGER,
            FOREIGN KEY(content_id) REFERENCES content_table(id),
            FOREIGN KEY(vss_rowid) REFERENCES vss_table(rowid)
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
    
    c.execute("SELECT vss_version()")
    print(c.fetchone())
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
        pages = loadPDF(file_path)
        chunks = chunkifyPDF(pages)
        for i in tqdm(range(len(chunks)), desc="Encoding chunks"):
            encoded.append(encodeChunks(model, chunks[i].page_content))
    elif (file_path.endswith(".md")):
        # tqdm.write(f"Indexing Markdown: {filename}")
        pages = loadText(file_path)
        chunks = chunkifyMarkdown(pages)
        for i in tqdm(range(len(chunks)), desc="Encoding chunks"):
            encoded.append(encodeChunks(model, chunks[i].page_content))
    elif (language := getProgrammingLanguage(os.path.splitext(file_path)[1])):
        # tqdm.write(f"Indexing Code: {filename}")
        pages = loadCode(file_path)
        chunks = chunkifyCode(pages, language)
        for i in tqdm(range(len(chunks)), desc="Encoding chunks"):
            encoded.append(encodeChunks(model, chunks[i].page_content))
    else:
        tqdm.write("Unsupported file type")
        return
    
    # tqdm.write("Inserting into database...")
    # c = conn.cursor()
    # c.execute("BEGIN TRANSACTION")
    # for i in tqdm(range(len(pages))):
    #     c.execute("INSERT INTO content_table (file_path, mod_time, metadata, content) VALUES (?, ?, ?, ?)", 
    #               (file_path, os.path.getmtime(file_path), json.dumps(pages[i].metadata), "".join(pages[i].page_content)))
    #     content_id = c.lastrowid
        
    #     for j in range(len(encoded[i])):
    #         c.execute("SELECT MAX(rowid) FROM vss_table")
    #         max_rowid = c.fetchone()[0]
    #         next_rowid = (max_rowid or 0) + 1
    #         c.execute("INSERT INTO vss_table (rowid, content_vector) VALUES (?, ?)", (next_rowid, encoded[i][j].tobytes()))
    #         c.execute("INSERT INTO id_mapping (content_id, vss_rowid) VALUES (?, ?)", (content_id, next_rowid))
    # # c.execute("COMMIT")
    # try:
    #     c.execute("COMMIT")
    # except sqlite3.OperationalError as e:
    #     print(f"ERROR: {e}")
    c = conn.cursor()

    content_data = []
    vss_data = []
    id_mapping_data = []

    # Get the initial max rowid
    c.execute("SELECT MAX(rowid) FROM vss_table")
    max_rowid = c.fetchone()[0] or 0

    for i in range(len(pages)):
        content_data.append((file_path, os.path.getmtime(file_path), json.dumps(pages[i].metadata), "".join(pages[i].page_content)))
        content_id = i + 1  # Assuming content_id starts at 1 and increments by 1 for each record
        
        for j in range(len(encoded[i])):
            max_rowid += 1
            vss_data.append((max_rowid, encoded[i][j].tobytes()))
            id_mapping_data.append((content_id, max_rowid))


    tqdm.write("Inserting into database...")
    c.execute("BEGIN TRANSACTION")
    c.executemany("INSERT INTO content_table (file_path, mod_time, metadata, content) VALUES (?, ?, ?, ?)", content_data)
    c.executemany("INSERT INTO vss_table (rowid, content_vector) VALUES (?, ?)", vss_data)
    c.executemany("INSERT INTO id_mapping (content_id, vss_rowid) VALUES (?, ?)", id_mapping_data)

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
    for file in tqdm(files, desc="Deleting file"):
        # delete row in content table
        c.execute("DELETE FROM content_table WHERE id = ?", (file[0],))
        c.execute("SELECT vss_rowid FROM id_mapping WHERE content_id = ?", (file[0],))
        vss_rowids = c.fetchall()
        for vss_rowid in vss_rowids:
            c.execute("DELETE FROM id_mapping WHERE content_id = ?", (file[0],))
            c.execute("DELETE FROM vss_table WHERE rowid = ?", (vss_rowid[0],))
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



def completeIndexDirectory(conn, directory_path : str):
    c = conn.cursor()

    total_files = count_files(directory_path)
    progress_bar = tqdm(total=total_files, desc=f"Indexing directory with {total_files} files")

    ignore_patterns = load_index_ignore(os.path.join(directory_path, ".indexignore"))

    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)

            if is_ignored(file_path, ignore_patterns):
                continue

            current_mode_time = os.path.getmtime(file_path)
            c.execute("SELECT mod_time FROM content_table WHERE file_path = ?", (file_path,))
            result = c.fetchone()
            if (result and current_mode_time - result[0] > 0.01):
                tqdm.write(f"File {file_path} has been modified. Reindexing...")
                deleteFile(conn, file_path)
                indexFile(conn, file_path)
            elif (not result):
                tqdm.write(f"File {file_path} is not indexed. Indexing...")
                indexFile(conn, file_path)
            else:
                tqdm.write(f"File {file_path} is already indexed. Skipping...")

            progress_bar.update()

    progress_bar.close()

    tqdm.write("Checking file existences...")
    c.execute("SELECT file_path FROM content_table")
    file_paths = c.fetchall()
    for file_path in tqdm(file_paths, desc="Checking file existence"):
        file_path = file_path[0]
        if not os.path.exists(file_path):
            tqdm.write(f"File {file_path} does not exist in the file system. Deleting...")
            deleteFile(conn, file_path)

    return

# todo: use scandir
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

def fts5Search(conn, query : str):
    c = conn.cursor()
    c.execute("SELECT * FROM fts_table WHERE content MATCH ? ORDER BY rank LIMIT 5", (query,))
    return c.fetchall()

def vssSearch(conn, query : str):

    sentences = sent_tokenize(query)
    encoded = encodeChunks(model, sentences)
    # find the average
    query_vector = np.mean(encoded, axis=0)

    c = conn.cursor()
    c.execute("""
        SELECT rowid, * FROM vss_table WHERE 
              vss_search(content_vector, 
                vss_search_params(?, 5)
              )
    """, (query_vector,))
    vectors_and_rows = set(c.fetchall())
    corresponding_documents = []

    for row in vectors_and_rows:
        c.execute("SELECT * FROM id_mapping WHERE vss_rowid = ?", (row[0],))
        doc_id = c.fetchone()[0]
        c.execute("SELECT * FROM content_table WHERE id = ?", (doc_id,))
        corresponding_documents.append(c.fetchone())

    # return reduceVssSearchResults(query_vector, corresponding_documents, 5)
    return corresponding_documents

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
    print("START")
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    if cuda.is_available():
        print("Using GPU")
        model = model.to('cuda')

    # conn = initDataBaseConnection("test_optimized_database.db")

    # c = conn.cursor()
    # c.execute("SELECT * FROM fts_table")
    # print(c.fetchall())

    # "C:\Users\docto\Documents\GitHub\uni-notes"

    # while True:
    #     query = input("Enter a query: ")
    #     if query == "exit":
    #         break
    #     results = fts5Search(conn, query)
    #     print(f"FTS5 Results ({len(results)}):")
    #     for result in results:
    #         print(result)

    # while True:
    #     query = input("Enter a query: ")
    #     if query == "exit":
    #         break
    #     results = vssSearch(conn, query)
    #     print(f"VSS Results ({len(results)}):")
    #     for result in results:
    #         print(result)

    # cProfile.run('indexDirectory(conn, "./tests")')
    # indexDirectory(conn, "../../uni-notes")
    # vacuum(conn)
    # optimize(conn)

    # indexDirectory(conn, "../../uni-notes")
    # completeIndexDirectory(conn, "../../uni-notes")

    print("Markdown...")
    md = loadText("./tests/module_6.md")
    # print(md)
    md_chunks = chunkifyMarkdown(md)
    print(md_chunks)
    md_chunks_encoded = encodeChunks(model, md_chunks[0].page_content)
    # print(md_chunks_encoded)

    print("PDF...")
    pdf = loadPDF("./tests/companion.pdf")
    # print(pdf)
    pdf_chunks = chunkifyPDF(pdf)
    print(pdf_chunks)
    pdf_chunks_encoded = encodeChunks(model, pdf_chunks[3].page_content)
    # print(pdf_chunks_encoded)

    print("Code...")
    code = loadCode("./test.py")
    language = getProgrammingLanguage(".py")
    # print(code)
    code_chunks = chunkifyCode(code, language)
    print(code_chunks)
    code_chunks_encoded = encodeChunks(model, code_chunks[0].page_content)
    # print(code_chunks_encoded)
    # conn.close()


def replace_latex_with_placeholder(text):
    # Regular expression pattern to match LaTeX expressions
    # latex_pattern = r'(?<!\\)\$(?=\S)(.*?\S)\$(?!\\)|\\\[.*?\\\]|\\\$\\\$.*?\\\$\\\$$'
    latex_pattern = r'(?<!\\)\$(\S.*?\S)\$(?!\\)|\\\[(.*?\S)\s*\\\]|\\\$\\\$([^$]*?\S)\s*\\\$\\\$'
    # Matches $$...$$, \[...\], and $...$
    # Requires contiguous non-whitespace characters between the dollar signs and the LaTeX content
    
    # Dictionary to store mappings between placeholders and original LaTeX expressions
    latex_mapping = {}
    
    # Counter for generating unique placeholders
    placeholder_counter = 0
    
    def replace_latex(match):
        nonlocal placeholder_counter
        placeholder = f'__LATEX_PLACEHOLDER_{placeholder_counter}__'
        for group in match.groups():
            if group is not None:
                latex_mapping[placeholder] = group  # Store mapping between placeholder and LaTeX expression
                break
        placeholder_counter += 1
        return placeholder
    
    # Replace LaTeX expressions with placeholders
    processed_text = re.sub(latex_pattern, replace_latex, text)
    
    return processed_text, latex_mapping


def restore_latex_placeholders(text, latex_mapping):
    # Replace placeholders with original LaTeX expressions
    for placeholder, latex_expression in latex_mapping.items():
        # print(placeholder, latex_expression)
        text = text.replace(placeholder, latex_expression)
    
    return text