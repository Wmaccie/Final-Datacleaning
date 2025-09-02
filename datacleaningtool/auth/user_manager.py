import bcrypt
from utils.logging_utils import get_db_connection


def authenticate_user(email, password):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()

    if user and bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        return {
            "email": user["email"],
            "is_admin": user["is_admin"],
            "id": user["id"]
        }
    return None


def create_user(email, password, is_admin=False):
    connection = get_db_connection()
    cursor = connection.cursor()
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    query = """
        INSERT INTO users (email, password_hash, is_admin)
        VALUES (%s, %s, %s)
    """
    cursor.execute(query, (email, hashed_pw, is_admin))
    connection.commit()
    cursor.close()
    connection.close()


def get_all_users():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT id, email, is_admin, created_at FROM users")
    users = cursor.fetchall()
    cursor.close()
    connection.close()
    return users


def reset_user_password(user_id, new_password):
    connection = get_db_connection()
    cursor = connection.cursor()
    hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    cursor.execute("UPDATE users SET password_hash = %s WHERE id = %s", (hashed_pw, user_id))
    connection.commit()
    cursor.close()
    connection.close()