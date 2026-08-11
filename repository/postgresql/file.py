import logging
from pathlib import Path
from repository.postgresql.common import get_postgres_client


logger = logging.getLogger(__name__)



INSERT_DOCUMENT_SOURCE_QUERY = """
    INSERT INTO document_source (name, file_path, content_summary)
    VALUES ($1, $2, $3)
    RETURNING id
"""
GET_FILE_SUMMARY_QUERY = """
    SELECT content_summary
    FROM document_source
    WHERE id = $1
"""



async def insert_document_source(name: str, file_path: Path, summary: str) -> int:
    """
        Inserts the given document source info into the document_source table and returns the created id.
    """
    try:
        pg_pool = get_postgres_client()

        async with pg_pool.acquire() as conn:
            doc_id = await conn.fetchval(INSERT_DOCUMENT_SOURCE_QUERY, name, str(file_path), summary)

        return doc_id
    except Exception as e:
        logger.error("insert_document_source : error inserting document source", extra={"error": str(e)})
        raise



async def get_file_summary(doc_id: int) -> str:
    """
        Fetches the content summary of the document with the given doc_id.
    """
    try:
        pg_pool = get_postgres_client()

        async with pg_pool.acquire() as conn:
            summary = await conn.fetchval(GET_FILE_SUMMARY_QUERY, doc_id)

        return summary
    except Exception as e:
        logger.error("get_file_summary : error fetching file summary", extra={"error": str(e)})
        raise
