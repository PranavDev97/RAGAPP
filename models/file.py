from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path


class FileSource(BaseModel):
    name: str = Field(description="The name of the file")
    file_path: Optional[Path] = Field(None, description="The path in which the file is saved") 
    content_bytes: bytes = Field(description="The content of the file in byte representation")
    content_summary: str = Field("", description="The summary of the file content")


        