
import database
import flask
import os
from config import DATABASE_NAME, LOCAL_DIRECTORY_PATH, SERVER_PORT

flask_app = flask.Flask(__name__)

@flask_app.route("/vss_search", methods=["GET"])
def vss_search():
    try:
        query = flask.request.args.get("query")
        if not query:
            flask.abort(400, description="Missing query parameter")
        results = db.vss_search(query)
    except Exception as e:
        flask.abort(500, description=f"Server error: {str(e)}")
    return flask.jsonify(results)

@flask_app.route("/fts5_search", methods=["GET"])
def fts5_search():
    try:
        query = flask.request.json.get("query")
        if not query:
            flask.abort(400, description="Missing query parameter")
        results = db.vss_search(query)
    except Exception as e:
        flask.abort(500, description=f"Server error: {str(e)}")
    return flask.jsonify(results)

@flask_app.route("/index",  methods=["POST"])
def index():
    try:
        path = flask.request.json.get("path", LOCAL_DIRECTORY_PATH)
        if not os.path.isdir(path):
            flask.abort(400, description="Invalid directory path")
        db.index_directory(path)
    except Exception as e:
        flask.abort(500, description=f"Server error: {str(e)}")
    return flask.jsonify({"message": f"Indexed directory {path}"})

@flask_app.route("/optimize",  methods=["POST"])
def optimize():
    try:
        db.optimize()
    except Exception as e:
        flask.abort(500, description=f"Server error: {str(e)}")
    
    return flask.jsonify({"message": "Optimized database"})

@flask_app.route("/vacuum",  methods=["POST"])
def vacuum():
    try:
        db.vacuum()
    except Exception as e:
        flask.abort(500, description=f"Server error: {str(e)}")
    
    return flask.jsonify({"message": "Vacuumed database"})

@flask_app.route("/reindex",  methods=["POST"])
def reindex():
    path = ""
    try:
        path = flask.request.json.get("path", LOCAL_DIRECTORY_PATH)
        if not os.path.isdir(path):
            flask.abort(400, description="Invalid directory path")
        db.reindex_directory(path)
    except Exception as e:
        flask.abort(500, description=f"Server error: {str(e)}")
    
    return flask.jsonify({"message": f"Reindexed directory {path}"})

if __name__ == "__main__":
    # okay so start the console application
    # flask_app.run(debug=True, port=SERVER_PORT)
    
    # db = database.VectorDatabase(DATABASE_NAME)
    # db.index_directory(LOCAL_DIRECTORY_PATH)

    db = database.VectorDatabase("OASDLKJASLKDJLAKSJDASD")
    # db.index_directory("./tests")

    while True:
        query = input("Enter query: ")
        results = db.hybrid_search(query, 5)
        # results = db.fts5_search(query)
        results = set(results)
        for result in results:
            print(result)
            # print(f"Similarity {result[0]}\n")
            # print(result[1])
            print("------------------------")