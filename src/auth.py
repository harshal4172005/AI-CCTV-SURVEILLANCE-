# src/auth.py

import sqlite3
import hashlib
import os

# --- FIX: Define a specific, consistent path for the database ---
# This ensures the main app and any other scripts use the same database file.
DB_PATH = os.path.join("app", "data", "users.db")

def hash_password(password):
    """Hashes the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_db_and_table():
    """Initializes the database and creates the users table if it doesn't exist."""
    # --- FIX: Ensure the 'app/data' directory exists before creating the DB ---
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
        print("First time setup: Creating default admin user...")
        try:
            admin_pass_hash = hash_password("admin123")
            cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                           ('admin', admin_pass_hash, 'Admin'))
            print("Default user 'admin' with password 'admin123' created.")
        except sqlite3.IntegrityError:
            print("Admin user already exists.")

    conn.commit()
    conn.close()

def verify_user(username, password):
    """Verifies user credentials and returns the user's role on success."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        password_hash, role = result
        if password_hash == hash_password(password):
            return role
    return None