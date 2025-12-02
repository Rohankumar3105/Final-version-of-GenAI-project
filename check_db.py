import sqlite3

conn = sqlite3.connect('data/telecom.db')
cursor = conn.cursor()

# Get customers table schema
cursor.execute('SELECT sql FROM sqlite_master WHERE type="table" AND name="customers"')
schema = cursor.fetchone()
if schema:
    print("Customers Table Schema:")
    print(schema[0])
    print("\n" + "="*50 + "\n")

# Get sample customer data
cursor.execute('SELECT * FROM customers LIMIT 3')
columns = [description[0] for description in cursor.description]
print("Columns:", columns)
print("\nSample Data:")
for row in cursor.fetchall():
    print(row)

conn.close()
