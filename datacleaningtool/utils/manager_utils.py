import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import bcrypt
import re
from typing import Dict, List, Optional, Any
import traceback
import numpy as np

# --- CONFIGURATIE & DATABASE CONNECTIE ---
# Laad de environment variabelen uit het .env bestand in de root van je project.
load_dotenv()

# Haal database credentials op uit de environment variabelen, met veilige defaults.
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "data_cleaner")

# --- DATABASE CONNECTIE & HASHING UTILITIES ---

def get_db_connection() -> Optional[mysql.connector.MySQLConnection]:
    """
    Maakt en retourneert een actieve MySQL database connectie.
    Retourneert None als de connectie mislukt.
    """
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        if connection.is_connected():
            return connection
    except Error as e_conn:
        # We printen de error naar de console voor debugging.
        # De UI-laag (bv. a_Login.py) is verantwoordelijk voor het tonen van een foutmelding aan de gebruiker.
        print(f"ERROR (manager_utils): Kan niet verbinden met MySQL: {e_conn}\n{traceback.format_exc()}")
    return None

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(plain_password: str, hashed_password_from_db: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password_from_db.encode('utf-8'))

# --- GEBRUIKERSAUTHENTICATIE & REGISTRATIE ---

def authenticate_user(email: str, password_provided: str) -> Optional[Dict[str, Any]]:
    """
    Authenticeert een gebruiker op basis van e-mail en wachtwoord.
    Retourneert een dictionary met gebruikersinformatie bij succes, anders None.

    Args:
        email: Het e-mailadres van de gebruiker.
        password_provided: Het door de gebruiker ingevoerde wachtwoord.

    Returns:
        Een dictionary met 'id', 'email', en 'is_admin' of None bij een fout of ongeldige login.
    """
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return None

        # 'with' statement zorgt ervoor dat de cursor automatisch wordt gesloten.
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT id, email, password_hash, is_admin FROM users WHERE email = %s", (email,))
            user_record = cursor.fetchone()

        # Verifieer of de gebruiker bestaat en of het wachtwoord klopt.
        if user_record and user_record["password_hash"] and verify_password(password_provided,
                                                                            user_record["password_hash"]):
            return {
                "id": user_record["id"],
                "email": user_record["email"],
                "is_admin": bool(user_record["is_admin"])
            }

        # Als gebruiker niet gevonden of wachtwoord incorrect
        return None

    except Error as e_auth:
        print(f"ERROR (manager_utils.authenticate_user) voor {email}: {e_auth}")
        return None
    finally:
        if conn and conn.is_connected():
            conn.close()


def create_user(email: str, password_provided: str, is_admin: bool = False) -> bool:
    """
    Registreert een nieuwe gebruiker in de database met een gehasht wachtwoord.
    Retourneert True bij succes. Werpt een ValueError bij logische fouten (bv. e-mail bestaat al).

    Args:
        email: Het e-mailadres voor het nieuwe account.
        password_provided: Het wachtwoord voor het nieuwe account.
        is_admin: Geeft aan of de nieuwe gebruiker een admin moet zijn.

    Returns:
        True als de gebruiker succesvol is aangemaakt.

    Raises:
        ValueError: Als het e-mailformaat ongeldig is, het wachtwoord te kort is,
                    of de gebruiker al bestaat.
    """
    # --- Input Validatie ---
    if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
        raise ValueError("Invalid email format.")
    if len(password_provided) < 8:
        raise ValueError("Password must be at least 8 characters long.")

    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            # In dit geval willen we een duidelijke fout, omdat de DB onbereikbaar is.
            raise ConnectionError("Could not connect to the database to create user.")

        with conn.cursor() as cursor:
            # Check of gebruiker al bestaat
            cursor.execute("SELECT email FROM users WHERE email = %s", (email,))
            if cursor.fetchone():
                raise ValueError(f"User with email '{email}' already exists.")

            # Hash het wachtwoord en voeg de nieuwe gebruiker toe
            hashed_pw_bytes = hash_password(password_provided)
            query = "INSERT INTO users (email, password_hash, is_admin, created_at) VALUES (%s, %s, %s, NOW())"

            cursor.execute(query, (email, hashed_pw_bytes.decode('utf-8'), is_admin))
            conn.commit()

        print(f"INFO (manager_utils.create_user): User '{email}' created successfully.")
        return True

    except Error as e_create:  # Vang specifieke database-errors op
        print(f"ERROR (manager_utils.create_user): Database error for {email}: {e_create}")
        # Vertaal een 'duplicate entry' error van de DB naar een leesbare ValueError
        if e_create.errno == 1062:
            raise ValueError(f"User with email '{email}' already exists (database constraint).")
        # Gooi andere DB errors opnieuw op als een algemene Exception
        raise Exception(f"A database error occurred: {e_create}") from e_create
    finally:
        if conn and conn.is_connected():
            conn.close()


# --- ADMIN-SPECIFIEKE GEBRUIKERSBEHEER FUNCTIES ---

def get_all_users() -> List[Dict[str, Any]]:
    """
    Haalt een lijst van alle gebruikers op uit de database, bedoeld voor de admin-pagina.
    Wachtwoord hashes worden uiteraard niet meegestuurd.

    Returns:
        Een lijst van dictionaries, waarbij elke dictionary een gebruiker representeert.
    """
    conn = None
    users_list: List[Dict[str, Any]] = []
    try:
        conn = get_db_connection()
        if conn is None:
            return []

        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT id, email, is_admin, created_at FROM users ORDER BY created_at DESC")
            raw_users = cursor.fetchall()
            for user_row in raw_users:
                # Zorg ervoor dat is_admin altijd een boolean is voor de frontend
                user_row['is_admin'] = bool(user_row.get('is_admin', False))
                users_list.append(user_row)
        return users_list

    except Error as e_get_all:
        print(f"ERROR (manager_utils.get_all_users): Kon gebruikers niet ophalen - {e_get_all}")
        return []  # Retourneer lege lijst bij fout
    finally:
        if conn and conn.is_connected():
            conn.close()


def reset_user_password(user_id: int, new_password: str) -> bool:
    """
    Reset het wachtwoord voor een specifieke gebruiker (admin only).

    Args:
        user_id: Het ID van de gebruiker wiens wachtwoord gereset moet worden.
        new_password: Het nieuwe, platte tekst wachtwoord.

    Returns:
        True als het resetten succesvol was, anders False.

    Raises:
        ValueError: Als het nieuwe wachtwoord niet aan de lengte-eis voldoet.
    """
    if len(new_password) < 8:
        raise ValueError("New password must be at least 8 characters long.")

    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False

        with conn.cursor() as cursor:
            hashed_pw_bytes = hash_password(new_password)
            cursor.execute("UPDATE users SET password_hash = %s WHERE id = %s",
                           (hashed_pw_bytes.decode('utf-8'), user_id))
            conn.commit()
            # cursor.rowcount > 0 controleert of er daadwerkelijk een rij is bijgewerkt.
            return cursor.rowcount > 0

    except Error as e_reset:
        print(f"ERROR (manager_utils.reset_user_password) voor user ID {user_id}: {e_reset}")
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()


def toggle_admin_status(user_id: int, new_admin_status: bool) -> bool:
    """
    Wijzigt de admin-status voor een specifieke gebruiker.

    Args:
        user_id: Het ID van de gebruiker.
        new_admin_status: De nieuwe status (True voor admin, False voor non-admin).

    Returns:
        True als de status succesvol is gewijzigd, anders False.
    """
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False

        with conn.cursor() as cursor:
            admin_flag = 1 if new_admin_status else 0
            cursor.execute("UPDATE users SET is_admin = %s WHERE id = %s", (admin_flag, user_id))
            conn.commit()
            return cursor.rowcount > 0

    except Error as e_toggle:
        print(f"ERROR (manager_utils.toggle_admin_status) voor user ID {user_id}: {e_toggle}")
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()


def delete_user_by_id(user_id: int) -> bool:
    """
    Verwijdert een gebruiker permanent uit de database op basis van ID.

    Args:
        user_id: Het ID van de te verwijderen gebruiker.

    Returns:
        True als de gebruiker succesvol is verwijderd, anders False.
    """
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False

        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
            conn.commit()
            deleted_count = cursor.rowcount
            print(f"INFO (manager_utils.delete_user_by_id): User ID {user_id} deleted. Rows affected: {deleted_count}")
            return deleted_count > 0

    except Error as e_delete:
        print(f"ERROR (manager_utils.delete_user_by_id) voor user ID {user_id}: {e_delete}")
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()


# --- AI USAGE LOGGING (voor de Cost Tracker) ---

def log_ai_usage(user_id: int, model: str, prompt_tokens: int, completion_tokens: int, cost: float):
    """Logt de details van een OpenAI API call in de database."""
    table_name = "api_usage_logs"
    conn = None
    try:
        conn = get_db_connection()
        if conn is None: return
        with conn.cursor() as cursor:
            query = f"INSERT INTO {table_name} (user_id, timestamp, model_used, prompt_tokens, completion_tokens, cost_usd) VALUES (%s, NOW(), %s, %s, %s, %s)"
            cursor.execute(query, (user_id, model, prompt_tokens, completion_tokens, cost))
            conn.commit()
    except Error as e_log:
        print(f"ERROR (log_ai_usage): Kon API usage niet loggen in DB: {e_log}")
    finally:
        if conn and conn.is_connected(): conn.close()

def get_ai_usage_stats() -> pd.DataFrame:
    """Haalt alle AI usage stats op uit de database."""
    table_name = "api_usage_logs"
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            query = f"SELECT u.email, l.timestamp, l.model_used, l.prompt_tokens, l.completion_tokens, l.cost_usd FROM {table_name} l JOIN users u ON l.user_id = u.id ORDER BY l.timestamp DESC"
            df = pd.read_sql(query, conn)
            return df
    except Error as e_stats:
        print(f"Error fetching AI usage stats: {e_stats}")
        raise e_stats
    finally:
        if conn and conn.is_connected(): conn.close()
    return pd.DataFrame()


def get_simulated_api_usage_logs(num_logs: int = 50) -> pd.DataFrame:
    """
    Generates a fake pandas DataFrame of API usage logs for demonstration.
    """
    # Fake data components
    users = ["whitney.m@qualogy.com", "test.user@example.com", "data.analyst@company.com", "demo@scrubhub.ai"]
    models = ["gpt-4o-mini", "gpt-5-nano", "gpt-4o-mini-planner"]

    data = {
        "timestamp": [datetime.now() - timedelta(hours=np.random.randint(0, 72),
                                                                   minutes=np.random.randint(0, 60)) for _ in
                      range(num_logs)],
        "email": [np.random.choice(users) for _ in range(num_logs)],
        "model_used": [np.random.choice(models) for _ in range(num_logs)],
        "prompt_tokens": [np.random.randint(500, 2000) for _ in range(num_logs)],
        "completion_tokens": [np.random.randint(50, 500) for _ in range(num_logs)],
        "cost_usd": [np.random.uniform(0.0001, 0.0025) for _ in range(num_logs)]
    }

    df = pd.DataFrame(data)

    # Sort by most recent timestamp first, like a real log
    df.sort_values(by="timestamp", ascending=False, inplace=True)

    return df

def toggle_admin_status(user_id: int, new_status: bool) -> bool:
    """Updates the admin status for a specific user."""
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE users SET is_admin = %s WHERE id = %s", (new_status, user_id))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        print(f"ERROR toggling admin status for user {user_id}: {e}")
        return False
    finally:
        if conn.is_connected(): conn.close()

def delete_user(user_id: int) -> bool:
    """Permanently deletes a user from the database."""
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn.cursor() as cursor:
            # Belangrijk: Overweeg wat er met gerelateerde data moet gebeuren (bv. api_usage_logs)
            # Voor nu verwijderen we alleen de gebruiker.
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        print(f"ERROR deleting user {user_id}: {e}")
        return False
    finally:
        if conn.is_connected(): conn.close()