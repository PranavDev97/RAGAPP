import logging
import asyncpg


logger = logging.getLogger(__name__)


# global postgres connection pool
pg_pool: asyncpg.Pool | None = None
# try to move the postgres client to fast api's state variable
async def create_postgres_client(
    user: str,
    password: str,
    database: str,
    host: str = "localhost",
    port: int = 5432,
    min_size: int = 1,
    max_size: int = 10,
) -> asyncpg.Pool:
    """
        Creates a postgres connection pool using the given credentials.
    """
    global pg_pool
    
    try:
        pg_pool = await asyncpg.create_pool(
            user=user,
            password=password,
            database=database,
            host=host,
            port=port,
            min_size=min_size,
            max_size=max_size,
        )

        return pg_pool
    except Exception as e:
        logger.error("create_postgres_client : error creating postgres client", extra={"error": str(e)})
        raise

def get_postgres_client() -> asyncpg.Pool:
    global pg_pool
    return pg_pool

