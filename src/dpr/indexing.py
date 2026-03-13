from psycopg2.extras import execute_values
from src.database.connection import get_connection
from src.dpr.encode import get_model

def index_passages(limit_passages=10_000, batch_size=128):
    conn = get_connection()
    cursor = conn.cursor()

    print(f"Récupération de {limit_passages} passages depuis la table 'passages'...")
    cursor.execute(
        """
        SELECT p.id, p.text
        FROM passages p
        LEFT JOIN dpr d ON d.passage_id = p.id
        WHERE d.passage_id IS NULL
        ORDER BY p.id
        LIMIT %s
        """,
        (limit_passages,)
    )
    rows = cursor.fetchall()
    passage_ids = [r[0] for r in rows]
    passage_texts = [r[1] for r in rows]

    if len(passage_ids) == 0:
        print("Tous les passages sont déjà encodés ou la table source est vide.")
    else:
        
        model = get_model()
        
        print(f"Encodage de {len(passage_ids)} passages en cours (cela peut prendre quelques minutes)...")
        embeddings = model.encode(
            passage_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print("Insertion des vecteurs dans la base de données...")
        data = [(int(pid), emb.tolist()) for pid, emb in zip(passage_ids, embeddings)]

        execute_values(
            cursor,
            """
            INSERT INTO dpr (passage_id, embedding)
            VALUES %s
            ON CONFLICT (passage_id) DO NOTHING
            """,
            data,
            template="(%s, %s::vector)",
            page_size=500
        )
        conn.commit()
        print(f"Terminé ! {len(data):,} embeddings ont été insérés.")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    
    index_passages(limit_passages=10_000, batch_size=128)