import logging
import redis.asyncio as redis


logger = logging.getLogger(__name__)


# global redis client
redis_client: redis.Redis | None = None

async def create_redis_client(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
) -> redis.Redis:
    """
        Creates a redis client using the given connection info.
    """
    global redis_client

    try:
        redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
        )
        await redis_client.ping()

        return redis_client
    except Exception as e:
        logger.error("create_redis_client : error creating redis client", extra={"error": str(e)})
        raise

def get_redis_client() -> redis.Redis:
    global redis_client
    return redis_client
