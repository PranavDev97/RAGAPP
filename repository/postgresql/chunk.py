import logging
from models.chunk import Chunk, ChunkType
from repository.postgresql.common import get_postgres_client


logger = logging.getLogger(__name__)


INSERT_CHUNK_QUERY = """
    INSERT INTO document_chunk (doc_id, chunk_index, chunk_text, chunk_type, chunk_summary, chunk_embedding, chunk_metadata)
    VALUES ($1, $2, $3, $4, $5, $6::vector, $7::jsonb)
"""
GET_CHUNKS_BY_SIMILARITY_QUERY = """
    WITH scored_chunks AS (
        SELECT
            chunk_index,
            chunk_text,
            chunk_metadata->>'name' AS chunk_name,
            1 - (chunk_embedding <=> $2::vector) AS score
        FROM 
            document_chunk
        WHERE 
            doc_id = $1
    ),
    threshold AS (
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY score) AS median_score
        FROM scored_chunks
    )
    SELECT
        sc.chunk_index,
        sc.chunk_text,
        sc.chunk_name,
        sc.score
    FROM 
        scored_chunks sc, threshold t
    WHERE 
        sc.score > t.median_score
    ORDER BY 
        sc.score DESC;
"""
GET_CHUNKS_BY_KEYWORD_QUERY = """
    WITH scored_chunks AS (
        SELECT
            chunk_index,
            chunk_text,
            chunk_metadata->>'name' AS chunk_name,
            ts_rank(chunk_tsvector, websearch_to_tsquery('english', $2)) AS score
        FROM 
            document_chunk
        WHERE 
            doc_id = $1
            AND chunk_tsvector @@ websearch_to_tsquery('english', $2)   
    )
    SELECT 
        chunk_index, 
        chunk_text,
        chunk_name,
        score
    FROM 
        scored_chunks
    WHERE 
        score > (
            SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY score)
            FROM scored_chunks
        )
    ORDER BY 
        score DESC;
"""



async def insert_chunks(doc_id: int, chunks: list[Chunk]) -> None:
    """
        Inserts the given chunks into the document_chunk table for the given doc_id.
    """
    try:
        rows = [
            (
                doc_id,
                index,
                chunk.table_content_markdown if chunk.chunk_type == ChunkType.TABLE else chunk.text_content,
                chunk.chunk_type.value,
                chunk.summary,
                str(chunk.embedding),
                chunk.metadata.model_dump_json() if chunk.metadata else "{}",
            )
            for index, chunk in enumerate(chunks)
        ]

        pg_pool = get_postgres_client()

        async with pg_pool.acquire() as conn:
            await conn.executemany(INSERT_CHUNK_QUERY, rows)
    except Exception as e:
        logger.error("insert_chunks : error inserting chunks", extra={"error": str(e)})
        raise



def _with_name(name: str | None, text: str) -> str:
    """
        Prepends the chunk name to the text.
    """
    if not name:
        return text

    return f"{name}\n{'-' * len(name)}\n{text}"



def _cutoff_by_score_gap(rows: list, score_gap: float) -> list:
    """
        Given rows sorted by score descending, returns the leading run of rows
        up to (but excluding) the first point where consecutive scores drop by
        more than score_gap.
    """
    for i in range(1, len(rows)):
        if rows[i - 1]["score"] - rows[i]["score"] > score_gap:
            return rows[:i]

    return rows



async def get_chunks_by_similarity(
    doc_id: int,
    vector: list[float],
    score_gap: float = 0.15,
) -> list[tuple[int, str, float]]:
    """
        Retrieves (chunk_index, chunk_text) for chunks of the given doc_id whose
        cosine similarity to the given vector exceeds min_similarity, then trims
        the result to the leading cluster of top scores: chunks are kept in
        descending score order until a consecutive score drop bigger than
        score_gap is found.
    """
    try:
        pg_pool = get_postgres_client()

        async with pg_pool.acquire() as conn:
            rows = await conn.fetch(
                GET_CHUNKS_BY_SIMILARITY_QUERY, doc_id, str(vector),
            )

        rows = _cutoff_by_score_gap(rows, score_gap)

        return [(row["chunk_index"], _with_name(row["chunk_name"], row["chunk_text"]), row["score"]) for row in rows]
    except Exception as e:
        logger.error("get_chunks_by_similarity : error retrieving chunks", extra={"error": str(e)})
        raise



async def get_chunks_by_keyword(
    doc_id: int,
    question: str,
    score_gap: float = 0.15,
) -> list[tuple[int, str, float]]:
    """
        Retrieves (chunk_index, chunk_text) for chunks of the given doc_id whose
        chunk_tsvector matches the given question via full-text keyword search
        and whose ts_rank exceeds min_rank, then trims the result to the leading
        cluster of top scores: chunks are kept in descending score order until a
        consecutive score drop bigger than score_gap is found.
    """
    try:
        pg_pool = get_postgres_client()

        async with pg_pool.acquire() as conn:
            rows = await conn.fetch(GET_CHUNKS_BY_KEYWORD_QUERY, doc_id, question)

        rows = _cutoff_by_score_gap(rows, score_gap)

        return [(row["chunk_index"], _with_name(row["chunk_name"], row["chunk_text"]), row["score"]) for row in rows]
    except Exception as e:
        logger.error("get_chunks_by_keyword : error retrieving chunks", extra={"error": str(e)})
        raise

