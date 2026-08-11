from fastapi import APIRouter, UploadFile, File
from models.file import FileSource
from services.chat import (
    upload_and_process as upload_and_process_svc,
    chat as chat_svc
)


chat_router = APIRouter()


@chat_router.post("/upload")
async def upload_and_process(file: UploadFile = File(...)):
    file_bytes = await file.read()
    file_source = FileSource(name=file.filename,content_bytes=file_bytes)

    doc_id = await upload_and_process_svc(file_source=file_source)

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "id":doc_id,
    }



@chat_router.get("/chat")
async def chat(question: str, doc_id: int):
    answer, chunk_meta = await chat_svc(
        question=question,
        doc_id=doc_id
    )

    return {
        "answer" : answer,
        "chunk meta" : chunk_meta
    }


