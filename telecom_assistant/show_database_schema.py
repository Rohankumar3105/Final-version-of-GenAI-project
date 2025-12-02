"""
Display Telecom Database Schema and Sample Data
================================================
Shows the complete database structure and sample records from each table
"""
import sqlite3
import os
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent / "data" / "telecom.db"

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)

def print_section(title):
    """Print section divider"""
    print("\n" + "-" * 100)
    print(f"  {title}")
    print("-" * 100)

def display_table_info(cursor, table_name):
    """Display table schema and sample data"""
    print_section(f"TABLE: {table_name}")
    
    # Get schema
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    
    print("\n📋 SCHEMA:")
    print("-" * 100)
    print(f"{'Column Name':<30} {'Type':<20} {'Not Null':<10} {'Default':<15}")
    print("-" * 100)
    for col in columns:
        col_id, col_name, col_type, not_null, default_val, pk = col
        not_null_str = "YES" if not_null else "NO"
        default_str = str(default_val) if default_val else "NULL"
        pk_marker = " (PK)" if pk else ""
        print(f"{col_name:<30} {col_type:<20} {not_null_str:<10} {default_str:<15}{pk_marker}")
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cursor.fetchone()[0]
    print(f"\n📊 Total Records: {row_count}")
    
    # Get sample data
    if row_count > 0:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        rows = cursor.fetchall()
        
        # Get column names
        col_names = [col[1] for col in columns]
        
        print("\n📝 SAMPLE DATA (First 5 records):")
        print("-" * 100)
        
        # Print column headers
        header = " | ".join([col[:18].ljust(18) for col in col_names])
        print(header)
        print("-" * 100)
        
        # Print rows
        for row in rows:
            row_str = " | ".join([str(val)[:18].ljust(18) if val is not None else "NULL".ljust(18) for val in row])
            print(row_str)
    else:
        print("\n⚠️  No data available in this table.")

def main():
    """Main function to display database info"""
    print_header("TELECOM DATABASE - SCHEMA AND SAMPLE DATA")
    
    if not os.path.exists(DB_PATH):
        print(f"\n❌ ERROR: Database not found at {DB_PATH}")
        return
    
    print(f"\n📂 Database Location: {DB_PATH}")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    
    print(f"\n📊 Total Tables: {len(tables)}")
    print("\nAvailable tables:")
    for i, table in enumerate(tables, 1):
        print(f"  {i}. {table[0]}")
    
    # Display each table
    for table in tables:
        table_name = table[0]
        display_table_info(cursor, table_name)
    
    # Summary
    print_header("DATABASE SUMMARY")
    
    print("\n📊 Table Statistics:")
    print("-" * 100)
    print(f"{'Table Name':<30} {'Row Count':<15} {'Column Count':<15}")
    print("-" * 100)
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        col_count = len(cursor.fetchall())
        print(f"{table_name:<30} {row_count:<15} {col_count:<15}")
    
    print("-" * 100)
    
    # Close connection
    conn.close()
    
    print_header("SCHEMA DISPLAY COMPLETED")
    print("\n✅ All tables displayed successfully!\n")

if __name__ == "__main__":
    main()
