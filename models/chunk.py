from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
from PIL import Image

class ChunkType(str, Enum):
    """Enum for the different types of content chunks."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"

class Chunk(BaseModel):
    chunk_type: ChunkType = Field(description="The type of content in the chunk.")
    text_content: Optional[str] = Field(None, description="The text content. An explanation of the content is provided for table and image chunks")
    table_content_markdown: Optional[str] = Field(None, description="The Markdown table representation for table chunks")
    image_content: Optional[Image.Image] = Field(None, description="The image embedded in the pdf")
    page_no: int = Field(description="The page number where the chunk originated.")
    bbox: Optional[list[float]] = Field(None, description="The bounding box [x0, y0, x1, y1] on the source page.")
    summary: Optional[str] = Field(None, description="The summary of the chunk content")
    embedding: Optional[list[float]] = Field(None, description="The vector embedding representation of the summary")

