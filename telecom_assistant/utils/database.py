# Database utilities
import sqlite3
import os

# Database path
db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'telecom.db')

def get_customer_by_id(customer_id):
    """
    Retrieve customer information by customer_id
    Returns: dict with customer info or None if not found
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT customer_id, name, email, phone_number as phone, 
                   service_plan_id as plan_id, account_status as status
            FROM customers 
            WHERE customer_id = ?
        """, (customer_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    except Exception as e:
        print(f"Database error: {e}")
        return None


def authenticate_customer(customer_id):
    """
    Authenticate a customer by customer_id
    Returns: dict with customer info if valid, None otherwise
    """
    # Check for admin login
    if customer_id.lower() == "admin":
        return {
            "customer_id": "admin",
            "name": "Administrator",
            "email": "admin@telecom.com",
            "phone": "N/A",
            "plan_id": "N/A",
            "status": "admin",
            "user_type": "admin"
        }
    
    # Regular customer authentication
    customer = get_customer_by_id(customer_id)
    if customer:
        customer['user_type'] = 'customer'
        return customer
    
    return None
