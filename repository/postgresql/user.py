import logging
from repository.postgresql.common import get_postgres_client


logger = logging.getLogger(__name__)



INSERT_USER_DETAILS_QUERY = """
    INSERT INTO users (username, password_hash)
    VALUES ($1, $2)
"""
INSERT_USER_DOCUMENT_QUERY = """
    INSERT INTO user_document (user_id, doc_id)
    VALUES ($1, $2)
"""
CHECK_USER_HAS_DOC_QUERY = """
    SELECT EXISTS (
        SELECT 1
        FROM user_document
        WHERE user_id = $1 AND doc_id = $2
    )
"""
GET_USER_DETAILS_QUERY="""
    SELECT
        id,
        password_hash
    FROM
        users
    WHERE
        username = $1
"""



async def insert_user_details(username: str, password_hash: str):
    """
        Inserts the username and password_hash to the table
    """
    try:
        pg_pool = get_postgres_client()

        async with pg_pool.acquire() as conn:
            await conn.execute(INSERT_USER_DETAILS_QUERY, username, password_hash)
    except Exception as e:
        logger.error("insert_user_details : error inserting user details", extra={"error": str(e)})
        raise



async def insert_user_document(user_id: int, doc_id:int):
    """
        Inserts the mapping information of user_id to doc_id
    """
    try:
        pg_pool = get_postgres_client()
    
        async with pg_pool.acquire() as conn:
            await conn.execute(INSERT_USER_DOCUMENT_QUERY, user_id, doc_id)
    except Exception as e:
        logger.error("insert_user_document : error inserting user document mapping", extra={"error": str(e)})
        raise



async def check_user_has_doc(user_id: int, doc_id: int) -> bool:
    """
        Checks whether there is a mapping between user_id and doc_id
    """
    try:
        pg_pool = get_postgres_client()

        async with pg_pool.acquire() as conn:
            return await conn.fetchval(CHECK_USER_HAS_DOC_QUERY, user_id, doc_id)
    except Exception as e:
        logger.error("check_user_has_doc : error checking user document mapping", extra={"error": str(e)})
        raise


async def get_user_details(username: str) -> tuple | None:
    """
        Gets users id and password hash by username
    """
    try:
        pg_pool = get_postgres_client()

        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(GET_USER_DETAILS_QUERY, username)

        return tuple(row) if row else None
    except Exception as e:
        logger.error("get_user_details : error fetching user details", extra={"error": str(e)})
        raise


