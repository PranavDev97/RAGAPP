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
INSERT_SESSION_INFO_QUERY = """
    INSERT INTO session (user_id, doc_id)
    VALUES ($1, $2)
    RETURNING session_id
"""
GET_SESSION_CHAT_QUERY = """
    SELECT chat
    FROM session_chat
    WHERE session_id = $1
    ORDER BY created_at DESC
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



async def get_session_chats(session_id: str) -> list[str]:
    """
        Fetches all the chats belonging to the given session
    """
    try:
        pg_pool = get_postgres_client()
        
        async with pg_pool.acquire() as conn:
            rows = await conn.fetch(GET_SESSION_CHAT_QUERY, session_id)
        
        return [row["chat"] for row in rows]
    except Exception as e:
        logger.error("get_session_chats : error fetching session chats", extra={"error": str(e)})
        raise

