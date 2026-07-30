from fastapi import APIRouter, UploadFile, File
from models.file import FileSource
from services.chat import upload_and_process as upload_and_process_svc

chat_router = APIRouter()

@chat_router.post("/upload")
async def upload_and_process(file: UploadFile = File(...)):
    file_bytes = await file.read()
    file_source = FileSource(name=file.filename,content=file_bytes)

    await upload_and_process_svc(file_source=file_source)

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
    }

