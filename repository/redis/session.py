import logging
import uuid
from repository.redis.common import get_redis_client


logger = logging.getLogger(__name__)


SESSION_KEY_PREFIX = "session:"
SESSION_CHAT_KEY_PREFIX = "session:chats:"



async def create_and_insert_session_info(user_id: int, doc_id: int) -> str:
    """
        Creates a new session id and stores the given user and document id against it.
    """
    try:
        redis_client = get_redis_client()

        session_id = str(uuid.uuid4())

        await redis_client.hset(
            f"{SESSION_KEY_PREFIX}{session_id}",
            mapping={"user_id": user_id, "doc_id": doc_id},
        )

        return session_id
    except Exception as e:
        logger.error("create_and_insert_session_info : error inserting session info", extra={"error": str(e)})
        raise



async def get_session_info(session_id: str) -> tuple[int, int]:
    """
        Fetches the user id and document id belonging to the given session id.
    """
    try:
        redis_client = get_redis_client()

        session_info = await redis_client.hgetall(f"{SESSION_KEY_PREFIX}{session_id}")

        return int(session_info["user_id"]), int(session_info["doc_id"])
    except Exception as e:
        logger.error("get_session_info : error fetching session info", extra={"error": str(e)})
        raise



async def add_chat_to_session(session_id: str, question: str, answer: str) -> None:
    """
        Appends the given chat message to the session's list of chats.
    """
    try:
        chat = f"Q. {question} \n A. {answer}"
        redis_client = get_redis_client()

        await redis_client.rpush(f"{SESSION_CHAT_KEY_PREFIX}{session_id}", chat)
    except Exception as e:
        logger.error("add_chat_to_session : error adding chat to session", extra={"error": str(e)})
        raise



async def get_session_chats(session_id: str) -> list[str]:
    """
        Fetches all the chats belonging to the given session, ordered oldest to newest.
    """
    try:
        redis_client = get_redis_client()

        return await redis_client.lrange(f"{SESSION_CHAT_KEY_PREFIX}{session_id}", 0, -1)
    except Exception as e:
        logger.error("get_session_chats : error fetching session chats", extra={"error": str(e)})
        raise
