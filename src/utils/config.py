import os
from dotenv import load_dotenv

def load_env_config():
    load_dotenv()
    config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'msmarco_db'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres'),
        'hf_token': os.getenv('HF_TOKEN', '')
    }
    
    return config