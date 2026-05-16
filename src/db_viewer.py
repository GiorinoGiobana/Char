import sqlite3
import argparse
import os
from tabulate import tabulate

def view_db(db_path):
    if not os.path.exists(db_path):
        print(f"Error: File {db_path} does not exist")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print(f"\nDatabase: {db_path}")
    print(f"Contains {len(tables)} tables: {[t[0] for t in tables]}")

    for table in tables:
        table_name = table[0]
        print(f"\n--- Table: {table_name} ---")
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Get first 5 rows of data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        rows = cursor.fetchall()
        
        if rows:
            print(tabulate(rows, headers=columns, tablefmt="grid"))
        else:
            print("(empty table)")
            
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple SQLite database viewer")
    parser.add_argument("db_path", help="Path to SQLite database file")
    args = parser.parse_args()
    
    view_db(args.db_path)
