from fastapi import APIRouter, UploadFile, File
from models.file import FileSource
from handler.auth.authorization import AutherizededUserId
from handler.auth.authentication import AuthenicatedUserId
from services.file import (
    upload_and_process as upload_and_process_svc,
    chat as chat_svc
)


file_router = APIRouter(prefix="/file")



@file_router.post("/upload")
async def upload_and_process(user_id: AuthenicatedUserId, file: UploadFile = File(...)):
    file_bytes = await file.read()
    file_source = FileSource(name=file.filename,content_bytes=file_bytes)

    session_id = await upload_and_process_svc(user_id=user_id, file_source=file_source)

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "session_id":session_id,
    }



@file_router.get("/chat")
async def chat(user_id: AutherizededUserId, question: str, session_id: str):
    answer, chunk_meta = await chat_svc(
        session_id=session_id,
        question=question
    )

    return {
        "answer" : answer,
        "chunk meta" : chunk_meta
    }