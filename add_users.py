# add_users.py

import sqlite3
import hashlib
import os

# --- FIX: Point to the same database path as the main app ---
DB_PATH = os.path.join("app", "data", "users.db")

# You can change the default passwords here if you like
USERS_TO_ADD = {
    "manager": ("manager123", "Manager"),
    "viewer": ("viewer123", "Viewer")
}

def hash_password(password):
    """Hashes the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password, role):
    """Adds a new user to the database."""
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at '{DB_PATH}'.")
        print("Please run the main Streamlit app first to create the database.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                       (username, password_hash, role))
        conn.commit()
        print(f"✅ Successfully added user: '{username}' with role: '{role}'")
    except sqlite3.IntegrityError:
        print(f"⚠️ User '{username}' already exists.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    print("Adding new users to the database...")
    for user, (pwd, user_role) in USERS_TO_ADD.items():
        add_user(user, pwd, user_role)
    print("\nScript finished.")