import logging
from models.chunk import Chunk, ChunkType
from repository.postgresql.common import get_postgres_client


logger = logging.getLogger(__name__)


INSERT_CHUNK_QUERY = """
    INSERT INTO document_chunk (doc_id, chunk_index, chunk_text, chunk_type, chunk_summary, chunk_embedding)
    VALUES ($1, $2, $3, $4, $5, $6::vector)
"""
GET_CHUNKS_BY_SIMILARITY_QUERY = """
    SELECT
        chunk_index,
        chunk_text,
        1 - (chunk_embedding <=> $2::vector) as score
    FROM document_chunk
    WHERE doc_id = $1
      AND 1 - (chunk_embedding <=> $2::vector) > $3
    ORDER BY score DESC
"""



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
    min_similarity: float = 0.5,
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
                GET_CHUNKS_BY_SIMILARITY_QUERY, doc_id, str(vector), min_similarity
            )

        rows = _cutoff_by_score_gap(rows, score_gap)

        return [(row["chunk_index"], row["chunk_text"], row["score"]) for row in rows]
    except Exception as e:
        logger.error("get_chunks_by_similarity : error retrieving chunks", extra={"error": str(e)})
        raise



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
            )
            for index, chunk in enumerate(chunks)
        ]

        pg_pool = get_postgres_client()

        async with pg_pool.acquire() as conn:
            await conn.executemany(INSERT_CHUNK_QUERY, rows)
    except Exception as e:
        logger.error("insert_chunks : error inserting chunks", extra={"error": str(e)})
        raise


