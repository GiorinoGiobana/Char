import sqlite3
import json
import os
import threading
from typing import List, Dict, Any, Union

def run_sqlite_query(db_path: str, query: str, timeout: float = 30.0) -> Union[List[Any], str]:
    """
    Execute SQLite query and return results.
    If execution fails, return result rows or error message string.

    Args:
        db_path: Database file path
        query: SQL query statement
        timeout: Timeout in seconds, default 30

    Returns:
        Query result list or error message string
    """
    if not os.path.exists(db_path):
        return f"Error: Database file not found, path: {db_path}"

    result_container = []
    error_container = []
    conn = None

    def _execute_query():
        nonlocal conn
        try:
            conn = sqlite3.connect(db_path, timeout=timeout)
            cursor = conn.cursor()
            cursor.execute(query)

            if query.strip().upper().startswith("SELECT") or query.strip().upper().startswith("PRAGMA"):
                results = cursor.fetchall()
                result_container.append(results)
            else:
                conn.commit()
                result_container.append("Success")

        except Exception as e:
            error_container.append(str(e))
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    thread = threading.Thread(target=_execute_query)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return f"Error: Query execution timeout ({timeout}s)"

    if error_container:
        return f"Error: {error_container[0]}"

    if result_container:
        return result_container[0]

    return f"Error: Unknown error"


def safe_remove_sandbox(db_path: str, recovery_json_path: str = None) -> None:
    """
    Safely delete sandbox database file and related journal files.
    """
    for path in [db_path, recovery_json_path]:
        if not path:
            continue
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    journal_path = f"{db_path}-journal" if db_path else None
    if journal_path:
        try:
            if os.path.exists(journal_path):
                os.remove(journal_path)
        except Exception:
            pass

def read_json_file(path: str) -> Dict[str, Any]:
    """Read JSON file and return its contents."""
    if not os.path.exists(path):
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {path} : {e}")
        return {}

def write_json_file(path: str, data: Any) -> bool:
    """Write data to JSON file."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error writing JSON file {path} : {e}")
        return False

def get_db_schema(db_path: str) -> str:
    """Get database schema using sqlite_master."""
    query = "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL;"
    results = run_sqlite_query(db_path, query)
    
    if isinstance(results, str) and results.startswith("Error"):
        return results
    
    schema_str = ""
    if results:
        for row in results:
            schema_str += row[0] + ";\n"
            
    return schema_str
