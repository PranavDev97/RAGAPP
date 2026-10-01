import logging
import hashlib
import jwt
from datetime import datetime, timedelta, timezone
from conf.config import JWT_SECRET_KEY, JWT_ALGORITHM
from repository.postgresql.user import insert_user_details, get_user_details
from repository.redis.session import create_and_insert_session_info


logger = logging.getLogger(__name__)


async def create_user(username: str, password: str):
    try:
        password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
        await insert_user_details(username=username, password_hash=password_hash)
    except Exception:
        logger.error("create_user : error creating user details")
        raise



async def validate_and_generate_token(username: str, password: str) -> str:
    try:
        user_details = await get_user_details(username=username)
    except Exception:  
        logger.error("login : error getting user details")
        raise

    if not user_details:
        raise Exception("user not found")

    user_id, actual_password_hash = user_details
    password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    if password_hash != actual_password_hash:
        raise Exception("incorrect password")

    jwt_payload = {
        "id":user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=1)
    }
    token = jwt.encode(jwt_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    return token



async def create_new_session(user_id: int, doc_id: int) -> str:
    try:
        session_id = await create_and_insert_session_info(
            user_id=user_id,
            doc_id=doc_id
        )
    except Exception:
        logger.error("create_new_session : error creating new session")
        raise

    return session_id


