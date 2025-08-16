# src/auth.py

import sqlite3
import hashlib
import os
import pandas as pd

# Define a specific, consistent path for the database
DB_PATH = os.path.join("app", "data", "users.db")

def hash_password(password):
    """Hashes the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_db_and_table():
    """Initializes the database and creates the users table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db_exists = os.path.exists(DB_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL
    )
    ''')
    
    if not db_exists:
        try:
            admin_pass_hash = hash_password("admin123")
            cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                           ('admin', admin_pass_hash, 'Admin'))
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()

def verify_user(username, password):
    """Verifies user credentials and returns the user's role on success."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    
    if result and result[0] == hash_password(password):
        return result[1]
    return None

def add_user(username, password, role):
    """Adds a new user to the database. Returns True on success, False on failure."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        password_hash = hash_password(password)
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                       (username, password_hash, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_all_users():
    """Retrieves all users from the database for display."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, username, role FROM users", conn)
    conn.close()
    return df

# --- NEW FUNCTION TO DELETE A USER ---
def delete_user(username):
    """Deletes a user from the database. Returns True on success."""
    # Safety check: do not delete the main admin account
    if username == 'admin':
        print("Error: Cannot delete the primary admin account.")
        return False
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    finally:
        conn.close()