import logging
from fastapi import APIRouter, HTTPException, Response
from models.user import CreateUserRequest, CreateSessionRequest
from services.user import (
    create_user as create_user_svc,
    validate_and_generate_token,
    create_new_session
)
from handler.auth.authorization import AuthenicatedUserId


logger = logging.getLogger(__name__)


user_router = APIRouter(prefix="/user")



@user_router.post("/register")
async def create_user(request: CreateUserRequest):
    request.username = request.username.strip()
    if not request.username:
        raise HTTPException(status_code=400, detail="username must not be empty")

    if not request.password or not request.password.strip():
        raise HTTPException(status_code=400, detail="password must not be empty")

    try:
        await create_user_svc(username=request.username, password=request.password)
    except:
        logger.error("create_user : error creating user")
        raise HTTPException(status_code=500, detail="Error creating user")

    return {
        "message": "User created successfully",
        "username": request.username,
    }



@user_router.post("/login")
async def login(request: CreateUserRequest, response: Response):
    request.username = request.username.strip()
    if not request.username:
        raise HTTPException(status_code=400, detail="username must not be empty")

    if not request.password or not request.password.strip():
        raise HTTPException(status_code=400, detail="password must not be empty")

    try:
        token = await validate_and_generate_token(username=request.username, password=request.password)
    except Exception as e:
        logger.error("login : error logging in", extra={"error": str(e)})
        raise HTTPException(status_code=401, detail="Invalid username or password")

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60,
    )

    return {
        "message": "Login successful",
        "username": request.username,
    }



@user_router.post("/logout")
async def logout(user_id: AuthenicatedUserId, response: Response):
    # TODO : add token blacklisting to avoid token usage of logged-out user

    response.set_cookie(
        key="access_token",
        value=None
    )

    return {
        "message" : "Logged-out successfully"
    }



@user_router.post("/session")
async def create_sesssion(user_id: AuthenicatedUserId, session_info : CreateSessionRequest):
    try:
        session_id = await create_new_session(
            user_id=user_id,
            doc_id=session_info.doc_id
        )
    except Exception:
        logger.error("create_sesssion : error creating session")
        raise HTTPException(status_code=500, detail="Error creating new session")

    return {
            "message": "Session Created successfully",
            "session_id":session_id,
    }
