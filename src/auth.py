#Username: admin
#Password: admin123

import sqlite3
import hashlib
import os

DB_FILE = "users.db"

def hash_password(password):
    """Hashes the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_db_and_table():
    """Initializes the database and creates the users table if it doesn't exist."""
    # Check if the database file exists. If not, we will create it and the admin user.
    db_exists = os.path.exists(DB_FILE)
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL
    )
    ''')
    
    # If the database was just created, add a default admin user
    if not db_exists:
        print("First time setup: Creating default admin user...")
        try:
            admin_pass_hash = hash_password("admin123")
            cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                           ('admin', admin_pass_hash, 'Admin'))
            print("Default user 'admin' with password 'admin123' created.")
        except sqlite3.IntegrityError:
            # This might happen in rare race conditions, but it's good practice to handle.
            print("Admin user already exists.")

    conn.commit()
    conn.close()

def verify_user(username, password):
    """Verifies user credentials and returns the user's role on success."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        password_hash, role = result
        # Verify the provided password against the stored hash
        if password_hash == hash_password(password):
            return role  # Login successful, return the user's role
    return None # Login failed