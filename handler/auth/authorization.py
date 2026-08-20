import logging
from typing import Annotated
from fastapi import HTTPException, Depends
from handler.auth.authentication import AuthenicatedUserId
from repository.postgresql.user import check_user_has_doc


logger = logging.getLogger(__name__)


async def authorize_user_doc(user_id: AuthenicatedUserId, doc_id: int | None = None) -> int:
    if doc_id is None:
        raise HTTPException(status_code=400, detail="Document ID required")

    try:
        has_doc = await check_user_has_doc(user_id=user_id, doc_id=doc_id)
    except Exception as e:
        logger.error("authorize_user_doc : error checking document ownership", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Error authorizing user")

    if not has_doc:
        raise HTTPException(status_code=403, detail="You do not have access to this document")

    return user_id

AutherizededUserId = Annotated[int, Depends(authorize_user_doc)]