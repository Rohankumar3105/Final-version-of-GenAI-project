import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'data', 'telecom.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("="*70)
print("DATABASE TABLES")
print("="*70)

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print("\nAvailable tables:")
for table in tables:
    print(f"  - {table[0]}")

print("\n" + "="*70)
print("TABLE SCHEMAS")
print("="*70)

for table in tables:
    table_name = table[0]
    print(f"\n{table_name}:")
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  {col[1]} ({col[2]})")

conn.close()
