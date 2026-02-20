import os
import psycopg2
from psycopg2 import Error
from psycopg2.extensions import connection
from .utils.config import load_env_config
from .utils.logger import get_logger

logger = get_logger(__name__)

def get_connection() -> connection:
    config = load_env_config()
    try:
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        logger.info("Database connection established successfully.")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        raise

def get_schema():
    # Load SQL schema from file
    schema_file_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    
    try:
        with open(schema_file_path, 'r') as f:
            schema_sql = f.read()
        logger.info("Database schema loaded successfully.")
        return schema_sql
    except FileNotFoundError as e:
        logger.error(f"Schema file not found: {e}")
        raise

def init_db() -> connection:
    # Load configuration
    config = load_env_config()

    # Check if database exists and create if it doesn't
    try:
        temp_conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database='postgres',
            user=config['user'],
            password=config['password']
        )
        temp_conn.autocommit = True
        temp_cursor = temp_conn.cursor()
        temp_cursor.execute(f"SELECT 1 FROM pg_database WHERE datname='{config['database']}'")
        exists = temp_cursor.fetchone()
        if not exists:
            logger.info(f"Database '{config['database']}' does not exist. Creating...")
            temp_cursor.execute(f"CREATE DATABASE {config['database']}")
            logger.info(f"Database '{config['database']}' created successfully.")
    except Exception as e:
        logger.error(f"Error checking/creating database: {e}")
        raise
    finally:
        temp_cursor.close()
        temp_conn.close()

    # Get connection
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Load and execute SQL schema
        schema_file_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        with open(schema_file_path, 'r') as f:
            schema_sql = f.read()

        logger.info("Executing database schema...")
        cursor.execute(schema_sql)
        conn.commit()
        logger.info("Database initialized successfully.")
    except Error as e:
        conn.rollback()
        logger.error(f"{e}")
        raise
    finally:
        cursor.close()
    
    return conn

def close_connection(conn: connection):
    try:
        conn.close()
        logger.info("Database connection closed successfully.")
    except Error as e:
        logger.error(f"Error closing the database connection: {e}")
        raise
    
if __name__ == "__main__":
    # Initialize the database
    conn = init_db()
    close_connection(conn)
    logger.info("Database initialization process completed.")