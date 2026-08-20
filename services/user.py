import logging
import hashlib
import jwt
from repository.postgresql.user import insert_user_details, get_user_details
from conf.config import JWT_SECRET_KEY, JWT_ALGORITHM
from datetime import datetime, timedelta, timezone


logger = logging.getLogger(__name__)


async def create_user(username: str, password: str):
    try:
        password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()

        await insert_user_details(username=username, password_hash=password_hash)

    except:
        logger.error("create_user : error creating user details")
        raise



async def validate_and_generate_token(username: str, password: str) -> str:
    try:
        user_details = await get_user_details(username=username)
    except:  
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



    

