from datasets import load_dataset
from psycopg2.extras import execute_values
from src.database.connection import get_connection
from src.utils.config import load_env_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def _extract_rows(data, number_of_rows: int):
    queries = {}
    passages = {}
    qrels = []

    n = len(data) if number_of_rows is None else min(len(data), number_of_rows)

    logger.info(f"Extracting rows from the dataset (total rows: {len(data)}, extracting: {n})...")
    for i in range(n):
        # Get the row
        row = data[i]

        # Extract query information
        query_id = row['query_id']
        query_text = row['query']
        queries[query_id] = query_text

        # Extract passage information
        passage_text = row['passages']['passage_text']
        passage_selected = row['passages']['is_selected']
        
        for text, rel in zip(passage_text, passage_selected):
            # Generate a new passage ID
            passage_id = len(passages) + 1
            passages[passage_id] = text

            qrels.append((query_id, passage_id, int(rel)))
    
    logger.info(f"Extracted {len(queries)} queries, {len(passages)} passages, and {len(qrels)} qrels.")
    return queries, passages, qrels

def populate_db():
    # Get complete dataset from Hugging Face
    logger.info("Loading dataset from Hugging Face...")
    dataset = load_dataset("microsoft/ms_marco", "v1.1")
    logger.info("Dataset loaded successfully.")

    # Extract rows from the training set
    queries, passages, qrels = _extract_rows(dataset['train'], 100000)

    # Connect to the database
    config = load_env_config()
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Insert queries into the database
        execute_values(
            cursor,
            """
            INSERT INTO queries (id, text) 
            VALUES %s
            ON CONFLICT (id) DO NOTHING
            """,
            [(query_id, query_text) for query_id, query_text in queries.items()],
            page_size=5000
        )
        logger.info(f"Inserting {len(queries)} queries into the database...")
        conn.commit()
        logger.info(f"Inserted {len(queries)} queries.")

        # Insert passages into the database
        execute_values(
            cursor,
            """
            INSERT INTO passages (id, text)
            VALUES %s
            ON CONFLICT (id) DO NOTHING
            """,
            [(passage_id, passage_text) for passage_id, passage_text in passages.items()],
            page_size=5000
        )
        logger.info(f"Inserting {len(passages)} passages into the database...")
        conn.commit()
        logger.info(f"Inserted {len(passages)} passages into the database.")

        # Insert qrels into the database in chunks to show progress and avoid long uncommitted transactions
        batch_size = 5000
        commit_every_batches = 10
        total_qrels = len(qrels)

        for start in range(0, total_qrels, batch_size):
            batch = qrels[start:start + batch_size]
            execute_values(
                cursor,
                """
                INSERT INTO qrels (query_id, passage_id, relevance)
                VALUES %s
                """,
                batch,
                page_size=batch_size
            )

            batch_index = (start // batch_size) + 1
            is_last_batch = start + batch_size >= total_qrels
            if batch_index % commit_every_batches == 0 or is_last_batch:
                conn.commit()
                inserted_so_far = min(start + batch_size, total_qrels)
                logger.info(f"Inserted {inserted_so_far}/{total_qrels} qrels into the database...")

        logger.info(f"Inserted {total_qrels} qrels into the database.")
        logger.info("Database population completed successfully.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error populating the database: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    populate_db()