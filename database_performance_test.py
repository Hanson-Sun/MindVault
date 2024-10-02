import sqlite3
import string
import random
import time
import numpy as np

def generate_random_string(length):
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for _ in range(length))
    return result_str

# ok performance test
conn = sqlite3.connect("test_performance.db")
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


c.execute("""
    CREATE TABLE IF NOT EXISTS content_table (
        id INTEGER PRIMARY KEY, 
        content TEXT 
    )
""")
# repeated insertion
random_string = generate_random_string(600)

start_time = time.time()

c.execute("BEGIN TRANSACTION;")
for i in range(10000):
    c.execute("INSERT INTO content_table(content) VALUES (?)", (random_string,))
    # conn.commit()
c.execute("COMMIT;")

end_time = time.time()

print("Time taken for 10000 insertions: ", end_time - start_time)
c.execute("DELETE FROM content_table;")
conn.commit()

# insertion with fts5 trigger
c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts_table USING fts5(content, content='content_table', content_rowid='id', tokenize='trigram case_sensitive 1')")
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

start_time = time.time()

c.execute("BEGIN TRANSACTION;")
for i in range(10000):
    c.execute("INSERT INTO content_table(content) VALUES (?)", (random_string,))
    # conn.commit()
c.execute("COMMIT;")

end_time = time.time()
print("Time taken for 10000 insertions with fts5 trigger: ", end_time - start_time)
c.execute("DELETE FROM content_table;")
conn.commit()

# repeate insertion in vss0 table
c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vss_table USING vss0(content_vector(1024))")

random_array = np.random.rand(1024)

start_time = time.time()
c.execute("BEGIN TRANSACTION;")
for i in range(10000):
    c.execute("INSERT INTO vss_table(rowid, content_vector) VALUES (?, ?)", (i, random_array,))
    # conn.commit()
c.execute("COMMIT;")
end_time = time.time()
c.execute("DELETE FROM vss_table;")
conn.commit()
print("Time taken for 10000 insertions in vss0 table: ", end_time - start_time)