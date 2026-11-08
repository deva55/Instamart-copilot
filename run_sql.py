import sqlite3
import pandas as pd
from pathlib import Path

# Define file paths
BASE_DIR = Path(__file__).resolve().parent

db_path = BASE_DIR / 'data' / 'instamart.db'
sql_path = BASE_DIR / 'sql' / 'queries.sql'

# Read SQL queries from file
with open(sql_path, 'r', encoding='utf-8') as file:
    sql = file.read()

# Split SQL queries and remove comment-only lines
queries = []

for statement in sql.split(';'):
    lines = [
        line for line in statement.splitlines()
        if line.strip() and not line.strip().startswith('--')
    ]

    query = '\n'.join(lines).strip()

    if query:
        queries.append(query)

# Connect and execute all queries
with sqlite3.connect(db_path) as conn:

    for i, query in enumerate(queries, start=1):
        result = pd.read_sql_query(query, conn)

        print(f"\n--- Query {i} ---")
        print(result.to_string(index=False))