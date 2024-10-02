import os

def start_console_app(database):
    while True:
        print("1. Search")
        print("2. Index")
        print("3. Optimize")
        print("4. Vacuum")
        print("5. Reindex")
        print("6. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            query = input("Enter search query: ")
            results = database.vss_search(query)
            print(results)
        elif choice == "2":
            path = input("Enter directory path to index: ")
            if not os.path.isdir(path):
                print("Invalid directory path")
                continue
            database.index_directory(path)
            print(f"Indexed directory {path}")
        elif choice == "3":
            database.optimize()
            print("Optimized database")
        elif choice == "4":
            database.vacuum()
            print("Vacuumed database")
        elif choice == "5":
            path = input("Enter directory path to reindex: ")
            if not os.path.isdir(path):
                print("Invalid directory path")
                continue
            database.reindex_directory(path)
            print(f"Reindexed directory {path}")
        elif choice == "6":
            break
        else:
            print("Invalid choice")