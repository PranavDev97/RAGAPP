from models.file import FileSource
from pathlib import Path
from processors.chunker import Chunk

UPLOAD_DIR = Path("files/chat")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

async def upload_and_process(file_source : FileSource):
    # Save file
    file_path = UPLOAD_DIR / file_source.name
    file_source.file_path = file_path

    with file_path.open("wb") as buffer:
        buffer.write(file_source.content)

    # Create chunks
    await Chunk(file_source=file_source)

    