import logging
from typing import Annotated
import jwt
from fastapi import Request, HTTPException, Depends
from conf.config import JWT_SECRET_KEY, JWT_ALGORITHM


logger = logging.getLogger(__name__)


async def authenticate_user(request: Request) -> int:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        logger.error("authenticate_user : invalid token", extra={"error": str(e)})
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = payload.get("id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    return user_id


AuthenicatedUserId = Annotated[int, Depends(authenticate_user)]
