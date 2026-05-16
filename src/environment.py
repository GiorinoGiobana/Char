import json
import os
import shutil
import sqlite3
import re
from typing import Tuple, Dict, Any, Optional, List
from .tools import run_sqlite_query, write_json_file, get_db_schema

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIRD_DEV_PATH = os.path.join(PROJECT_ROOT, "bird", "dev")
DEV_MODIFIED_JSON = os.path.join(BIRD_DEV_PATH, "dev_modified.json")
DEV_DATABASES_DIR = os.path.join(BIRD_DEV_PATH, "dev_databases")
TEMP_ENV_DIR = os.path.join(PROJECT_ROOT, "temp_env")

def setup_sandbox(question: str, temp_dir: str = None) -> Tuple[str, str]:
    """
    Set up sandbox environment for the given question.
    Returns: (db_path, recovery_json_path)
    """
    if temp_dir is None:
        temp_dir = TEMP_ENV_DIR

    # 1. Match configuration
    config = _find_config_by_question(question)
    if not config:
        raise ValueError(f"Question not found in {DEV_MODIFIED_JSON} ")
    
    db_id = config['db_id']
    modified_sql = config.get('modified', '')
    gold_sql = config.get('SQL', '')
    question_id = config['question_id']
    
    # 2. Create sandbox
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    original_db_path = os.path.join(DEV_DATABASES_DIR, db_id, f"{db_id}.sqlite")
    if not os.path.exists(original_db_path):
         pass

    sandbox_db_path = os.path.join(temp_dir, f"{question_id}_{db_id}.sqlite")
    
    try:
        shutil.copy2(original_db_path, sandbox_db_path)
    except FileNotFoundError:
         raise FileNotFoundError(f"Question not found in {original_db_path} ")

    recovery_json_path = os.path.join(temp_dir, f"{question_id}_recovery.json")
    recovery_context: Dict[str, Any] = {}

    # 3. Dynamic recovery context generation and execution
    if modified_sql:
        print(f"Applying modification: {modified_sql}")
        recovery_context = _generate_context_and_modify(sandbox_db_path, modified_sql, gold_sql)
    
    # Save recovery context: do not generate empty file when no damage, for Router to determine CASE_2_NORMAL
    if recovery_context:
        write_json_file(recovery_json_path, recovery_context)
    else:
        try:
            if os.path.exists(recovery_json_path):
                os.remove(recovery_json_path)
        except Exception:
            pass
    
    return sandbox_db_path, recovery_json_path

def _find_config_by_question(question: str) -> Optional[Dict]:
    try:
        with open(DEV_MODIFIED_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if item['question'].strip() == question.strip():
                    return item
    except Exception as e:
        print(f"Error reading configuration: {e}")
    return None

def _get_primary_key(cursor, table_name: str) -> str:
    """Get table primary key, return rowid if no explicit primary key"""
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        # columns format: (cid, name, type, notnull, dflt_value, pk)
        pks = [col[1] for col in columns if col[5] > 0]
        if len(pks) == 1:
            return pks[0]
        elif len(pks) > 1:
            return pks[0] # Simplified handling: for composite primary keys, only take the first one for now
        else:
            return "rowid"
    except:
        return "rowid"

def _quote_sqlite_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f"\"{escaped}\""


def _parse_top_level_clauses(sql: str) -> Dict[str, str]:
    """
    Roughly parse top-level SQL clauses (not covering all SQL syntax, only BIRD/Spider common styles).
    Returned keys only include clauses that appeared: from/where/group_by/having/order_by/limit
    """
    clause_keywords = [
        ("GROUP BY", "group_by"),
        ("ORDER BY", "order_by"),
        ("HAVING", "having"),
        ("WHERE", "where"),
        ("FROM", "from"),
        ("LIMIT", "limit"),
    ]

    positions: List[Tuple[int, str, str]] = []
    depth = 0
    quote: Optional[str] = None  # "'", '"', '`'
    i = 0
    sql_len = len(sql)
    while i < sql_len:
        ch = sql[i]

        if quote is not None:
            if ch == quote:
                if quote == "'" and i + 1 < sql_len and sql[i + 1] == "'":
                    i += 2
                    continue
                quote = None
            i += 1
            continue

        if ch in ("'", '"', '`'):
            quote = ch
            i += 1
            continue

        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            i += 1
            continue

        if depth == 0 and ch.isalpha():
            for kw, key in clause_keywords:
                kw_len = len(kw)
                if i + kw_len <= sql_len and sql[i:i + kw_len].upper() == kw:
                    before_ok = i == 0 or not (sql[i - 1].isalnum() or sql[i - 1] == "_")
                    after_ok = i + kw_len == sql_len or not (sql[i + kw_len].isalnum() or sql[i + kw_len] == "_")
                    if before_ok and after_ok:
                        positions.append((i, kw, key))
                        i += kw_len
                        break
            else:
                i += 1
            continue

        i += 1

    positions.sort(key=lambda x: x[0])
    boundaries: List[Tuple[str, int, int]] = []
    for idx, (_pos, _kw, key) in enumerate(positions):
        start = _pos + len(_kw)
        end = positions[idx + 1][0] if idx + 1 < len(positions) else sql_len
        boundaries.append((key, start, end))

    clauses: Dict[str, str] = {}
    for key, start, end in boundaries:
        clauses[key] = sql[start:end].strip().rstrip(";")
    return clauses


def _extract_table_alias(from_clause: str, table_name: str) -> str:
    """
    Question not found in FROM/JOIN Find an alias for the specified table in FROM/JOIN clause (if no explicit alias, return table_name).
    """
    tokens_that_are_not_alias = {
        "ON", "USING", "INNER", "LEFT", "RIGHT", "FULL", "CROSS",
        "JOIN", "WHERE", "GROUP", "ORDER", "LIMIT", "HAVING",
    }

    main_table_pattern = re.compile(
        r"(?is)^\s*([`\"\[]?[A-Za-z_][\w\.]*[`\"\]]?)\s*(?:AS\s+)?([A-Za-z_]\w*)?",
    )
    m_main = main_table_pattern.search(from_clause)
    if m_main:
        raw_table = m_main.group(1) or ""
        raw_table_stripped = raw_table.strip('`"[]')
        if raw_table_stripped == table_name:
            alias = (m_main.group(2) or "").strip()
            if alias and alias.upper() not in tokens_that_are_not_alias:
                return alias

    join_pattern = re.compile(
        r"(?is)\bJOIN\s+([`\"\[]?[A-Za-z_][\w\.]*[`\"\]]?)\s*(?:AS\s+)?([A-Za-z_]\w*)?",
    )
    for m in join_pattern.finditer(from_clause):
        raw_table = m.group(1) or ""
        raw_table_stripped = raw_table.strip('`"[]')
        if raw_table_stripped != table_name:
            continue

        alias = (m.group(2) or "").strip()
        if not alias or alias.upper() in tokens_that_are_not_alias:
            return table_name
        return alias

    return table_name


def _scan_columns_for_alias(sql: str, alias: str) -> List[str]:
    """
    Scan column references in the form alias.col / alias.`col` / alias.\"col\" / alias[ col ].
    Return deduplicated column name list (maintaining stable order).
    """
    pattern = re.compile(
        rf"(?is)\b{re.escape(alias)}\s*\.\s*(?:`([^`]+)`|\"([^\"]+)\"|\[([^\]]+)\]|([A-Za-z_]\w*))"
    )
    seen = set()
    cols: List[str] = []
    for m in pattern.finditer(sql):
        col = next((g for g in m.groups() if g), "")
        col = col.strip()
        if not col or col in seen:
            continue
        seen.add(col)
        cols.append(col)
    return cols


def _scan_unqualified_quoted_identifiers(sql: str) -> List[str]:
    pattern = re.compile(r"(?s)(?:`([^`]+)`|\"([^\"]+)\"|\[([^\]]+)\])")
    seen = set()
    tokens: List[str] = []
    for m in pattern.finditer(sql):
        token = next((g for g in m.groups() if g), "")
        token = token.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _get_table_columns(cursor: sqlite3.Cursor, table_name: str) -> List[str]:
    cols: List[str] = []
    try:
        cursor.execute(f"PRAGMA table_info({_quote_sqlite_identifier(table_name)})")
        rows = cursor.fetchall()
        for r in rows:
            cols.append(r[1])
    except Exception:
        pass
    return cols


def _required_not_null_columns(cursor: sqlite3.Cursor, table_name: str) -> List[str]:
    required_cols: List[str] = []
    try:
        cursor.execute(f"PRAGMA table_info({_quote_sqlite_identifier(table_name)})")
        rows = cursor.fetchall()
        for r in rows:
            # r: (cid, name, type, notnull, dflt_value, pk)
            name = r[1]
            notnull = r[3]
            dflt = r[4]
            if notnull == 1 and dflt is None:
                required_cols.append(name)
    except Exception:
        pass
    return required_cols


def _build_minimal_payload_for_table(
    cursor: sqlite3.Cursor,
    gold_sql: str,
    table_name: str,
    extra_columns: Optional[List[str]] = None,
    min_topk: int = 10,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Based on gold SQL, extract minimal required columns + rows for the target table from the original database.
    Return (payload_columns, data_payload_rows)
    """
    if not gold_sql:
        return [], []

    clauses = _parse_top_level_clauses(gold_sql)
    from_clause = clauses.get("from")
    if not from_clause:
        return [], []

    alias = _extract_table_alias(from_clause, table_name)
    referenced_cols = _scan_columns_for_alias(gold_sql, alias)

    # Single-table queries often use unqualified column names (without alias.), use "quoted identifier \u2229 table field set" for supplementary extraction.
    if not referenced_cols and not re.search(r"(?is)\bJOIN\b", from_clause):
        table_cols = set(_get_table_columns(cursor, table_name))
        for tok in _scan_unqualified_quoted_identifiers(gold_sql):
            if tok in table_cols:
                referenced_cols.append(tok)

    required_cols = []
    seen = set()
    for c in (referenced_cols + (extra_columns or []) + _required_not_null_columns(cursor, table_name)):
        if c and c not in seen:
            required_cols.append(c)
            seen.add(c)

    if not required_cols:
        pk = _get_primary_key(cursor, table_name)
        if pk != "rowid":
            required_cols = [pk]
        else:
            table_cols = _get_table_columns(cursor, table_name)
            if table_cols:
                required_cols = [table_cols[0]]
            else:
                return [], []

    select_items = [f"{alias}.{_quote_sqlite_identifier(c)} AS {_quote_sqlite_identifier(c)}" for c in required_cols]
    select_sql = "SELECT DISTINCT " + ", ".join(select_items) + f" FROM {from_clause}"

    where_clause = clauses.get("where")
    order_by_clause = clauses.get("order_by")
    limit_clause = clauses.get("limit")

    if where_clause:
        select_sql += f" WHERE {where_clause}"

    if order_by_clause and (limit_clause or not where_clause):
        select_sql += f" ORDER BY {order_by_clause}"

    if limit_clause:
        try:
            base_limit = int(re.findall(r"\d+", limit_clause)[0])
        except Exception:
            base_limit = min_topk
        effective_limit = max(base_limit, min_topk)
        select_sql += f" LIMIT {effective_limit}"
    elif order_by_clause and not where_clause:
        select_sql += f" LIMIT {min_topk}"

    cursor.execute(select_sql)
    rows = cursor.fetchall()
    return required_cols, [dict(r) for r in rows]


def _generate_context_and_modify(db_path: str, sql: str, gold_sql: str = "") -> Dict[str, Any]:
    """
    Analyze destructive SQL, save recovery data, and execute SQL.
    Strictly follow keypoint.md design to generate recovery_context.json
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row # Convenient for dict conversion
    cursor = conn.cursor()
    context = {}
    
    # Normalize SQL
    sql_stripped = sql.strip()
    sql_upper = sql_stripped.upper()
    
    # SQL to be executed (may need conversion)
    executable_sql = sql_stripped
    
    try:
        # Case 1: DROP TABLE (drop table)
        if sql_upper.startswith("DROP TABLE"):
            match = re.search(r"DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?([^\s;]+)", sql_stripped, re.IGNORECASE)
            if match:
                table_name = match.group(1).strip('`"[]')
                
                context['case_type'] = 'missing_table'
                context['target_table'] = table_name
                
                # Save Schema
                cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                schema_row = cursor.fetchone()
                if schema_row:
                    context['table_schema'] = schema_row['sql']

                try:
                    payload_columns, payload_rows = _build_minimal_payload_for_table(
                        cursor=cursor,
                        gold_sql=gold_sql,
                        table_name=table_name,
                        extra_columns=[],
                    )
                    if payload_columns and payload_rows:
                        context['payload_columns'] = payload_columns
                        context['data_payload'] = payload_rows
                    else:
                        cursor.execute(f"SELECT * FROM {_quote_sqlite_identifier(table_name)}")
                        rows = cursor.fetchall()
                        context['data_payload'] = [dict(row) for row in rows]
                except Exception as e:
                    print(f"Warning: Cannot compactly backup DROP TABLE data by gold SQL, falling back to full table backup: {e}")
                    cursor.execute(f"SELECT * FROM {_quote_sqlite_identifier(table_name)}")
                    rows = cursor.fetchall()
                    context['data_payload'] = [dict(row) for row in rows]

        # Case 3: UPDATE (data corruption)
        elif sql_upper.startswith("UPDATE"):
            # Improved Regex: capture table name after UPDATE, column names after SET (simplified), and WHERE
            match = re.search(r"UPDATE\s+(.+?)\s+SET\s+(.+?)\s+WHERE\s+(.+)", sql_stripped, re.IGNORECASE | re.DOTALL)
            if match:
                table_name = match.group(1).strip('`"[]')
                set_clause = match.group(2).strip()
                condition = match.group(3).strip().rstrip(';')
                
                pk = _get_primary_key(cursor, table_name)
                
                # Parse set_clause to get affected columns
                # Simple parsing: assume format is col = val
                # Actually could be col1=v1, col2=v2
                # Try to extract column names
                columns_to_fix: List[str] = []
                col_pattern = re.compile(r"(?is)(?:^|,)\s*(?:`([^`]+)`|\"([^\"]+)\"|\[([^\]]+)\]|([A-Za-z_]\w*))\s*=")
                for m_col in col_pattern.finditer(set_clause):
                    col = next((g for g in m_col.groups() if g), "")
                    col = col.strip().strip('`"[]')
                    if col and col not in columns_to_fix:
                        columns_to_fix.append(col)
                
                context['case_type'] = 'data_corruption'
                context['target_table'] = table_name
                context['primary_key'] = pk
                context['columns_to_fix'] = columns_to_fix
                
                # Key optimization: only query PK and modified columns, and only for rows matching WHERE condition
                try:
                    if pk == "rowid":
                        col_exprs = ["rowid AS " + _quote_sqlite_identifier("rowid")]
                    else:
                        col_exprs = [_quote_sqlite_identifier(pk) + " AS " + _quote_sqlite_identifier(pk)]

                    for c in columns_to_fix:
                        col_exprs.append(_quote_sqlite_identifier(c) + " AS " + _quote_sqlite_identifier(c))

                    select_query = (
                        "SELECT " + ", ".join(col_exprs)
                        + f" FROM {_quote_sqlite_identifier(table_name)} WHERE {condition}"
                    )
                    
                    cursor.execute(select_query)
                    rows = cursor.fetchall()
                    
                    context['data_payload'] = [dict(row) for row in rows]
                    
                except Exception as e:
                    print(f"Warning: Cannot save UPDATE backup: {e}")

        # Case 4: DROP COLUMN (drop column)
        elif sql_upper.startswith("DROP FIELD") or "DROP COLUMN" in sql_upper:
            table_name = ""
            col_name = ""
            
            # Try to match "DROP Field table.col"
            match_pseudo = re.search(r"DROP\s+Field\s+([^\.]+)\.(.+)", sql_stripped, re.IGNORECASE)
            if match_pseudo:
                table_name = match_pseudo.group(1).strip('`"[]')
                col_name = match_pseudo.group(2).strip('`"[]')
            else:
                match_alter = re.search(r"ALTER\s+TABLE\s+([^\s]+)\s+DROP\s+COLUMN\s+([^\s;]+)", sql_stripped, re.IGNORECASE)
                if match_alter:
                    table_name = match_alter.group(1).strip('`"[]')
                    col_name = match_alter.group(2).strip('`"[]')
            
            if table_name and col_name:
                executable_sql = f"ALTER TABLE {_quote_sqlite_identifier(table_name)} DROP COLUMN {_quote_sqlite_identifier(col_name)}"
                pk = _get_primary_key(cursor, table_name)
                
                context['case_type'] = 'missing_column'
                context['target_table'] = table_name
                context['primary_key'] = pk
                context['columns_to_fix'] = [col_name]
                
                try:
                    extra_cols = [pk] if pk != "rowid" else []
                    payload_columns, payload_rows = _build_minimal_payload_for_table(
                        cursor=cursor,
                        gold_sql=gold_sql,
                        table_name=table_name,
                        extra_columns=list(dict.fromkeys(extra_cols + [col_name])),
                    )
                    if payload_columns and payload_rows:
                        context['payload_columns'] = payload_columns
                        context['data_payload'] = payload_rows
                    else:
                        # Fallback: full table backup (only PK + dropped column)
                        if pk == "rowid":
                            cols_str = "rowid AS " + _quote_sqlite_identifier("rowid") + ", " + _quote_sqlite_identifier(col_name) + " AS " + _quote_sqlite_identifier(col_name)
                        else:
                            cols_str = (
                                _quote_sqlite_identifier(pk) + " AS " + _quote_sqlite_identifier(pk)
                                + ", " + _quote_sqlite_identifier(col_name) + " AS " + _quote_sqlite_identifier(col_name)
                            )
                        cursor.execute(f"SELECT {cols_str} FROM {_quote_sqlite_identifier(table_name)}")
                        rows = cursor.fetchall()
                        context['data_payload'] = [dict(row) for row in rows]
                except Exception as e:
                     print(f"Warning: Cannot save DROP COLUMN backup: {e}")

        # Execute destructive SQL
        print(f"Execute SQL: {executable_sql}")
        cursor.execute(executable_sql)
        conn.commit()
        
    except Exception as e:
        print(f"Error during environment modification: {e}")
    finally:
        conn.close()
        
    return context
