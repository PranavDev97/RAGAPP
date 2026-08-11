import asyncio
import logging
import mimetypes
from pathlib import Path
from typing import Optional
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types
from models.chunk import Chunk, ChunkType
from models.file import FileSource


logger = logging.getLogger(__name__)


# constants
CHEAP_MODEL = "gemini-3.5-flash-lite"


FILE_SUMMARY_SYSTEM_INSTRUCTION = """
You are provided with a full document. Summarize the entire file so as to understand the concepts in the file and also include key entities (such as people, institutions, concepts, or objects) 
present in the file content, and briefly explain their specific role, relevance, and connection to the events or ideas described in the file.
"""
CHUNK_SUMMARY_SYSTEM_INSTRUCTION = """
You are provided with a full document and a specific chunk extracted from it. Your task is to generate a short, succinct context that situates the chunk within the overall document to improve search retrieval.

Identify key entities (such as people, institutions, concepts, or objects) present in the chunk, and briefly explain their specific role, relevance, and connection to the events or ideas described in that chunk.

Limit your response to 600 tokens. Output only this contextual summary and entity explanations as single paragraph in plain text — do not include any additional commentary or introductory text.
"""
IMAGE_DESCRIPTION_SYSTEM_INSTRUCTION = """
You are given an image. Create a detailed description of the image such that no details of the image is missed. Always include any text inside the image in the decription being generated and explain its usage within the image.
"""
SEARCH_STATEMENT_SYSTEM_INSTRUCTION="""
You are given a question and a file summary. Convert the question into a statement in plaint text using the file summary as a context for better semantic search retrieval of valid chunks of the same document.
"""
DOCUMENT_ANSWERING_SYSTEM_INSTRUCTION = """
You are given a question and the relevant chunk contents from a document. Generate the most apt answer to the question in simple words.
"""


# global gemini client
ai_client: genai.Client | None = None
async def create_gemini_client(api_key : str, http_opts: types.HttpOptions) -> genai.Client:
    """
        Creates a gemini client using the given api key and http options.
    """
    global ai_client
    ai_client = genai.Client(api_key=api_key, http_options=http_opts)

            

async def _create_cached_file_content(model: str, system_prompt: str, file_path : Path, cache_id : str):
    """
        Creates a cached file content
    """
    # upload file
    try:
        f = ai_client.files.upload(file=file_path)
        # wait till processing completed
        while f.state.name == "PROCESSING":
            await asyncio.sleep(3)
            f = ai_client.files.get(name=f.name)
    except Exception as e:
        logger.error("_create_cached_file_content : error uploading file",extra={"error": str(e)})
        raise

    # create cache using the uploaded file content            
    try:
        cache = ai_client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                display_name=cache_id,
                system_instruction=system_prompt,
                contents=[f],
                ttl="600s",
            )
        )

        return cache.name
    except Exception as e:
        logger.error("_create_cached_file_content : error creating cache",extra={"error": str(e)})
        raise



async def _generate_content(model: str, messages : list[types.Content],  
                            system_instruction: Optional[str]=None, cache_name: Optional[str]=None
) -> str:
    """
        Generates AI content based on the input messages given
    """
    config_params = {
        "temperature": 0.2,
        "max_output_tokens": 10000,
    }

    if cache_name:
        config_params["cached_content"] = cache_name
    if system_instruction:
        config_params["system_instruction"] = system_instruction

    try:
        response = ai_client.models.generate_content(
            model=model,
            contents=messages,
            config=types.GenerateContentConfig(**config_params)
        )

        return response.text
    except Exception as e:
        logger.error("_generate_content : error generating content", extra={"error": str(e)})
        raise



async def summarize_file(file_source: FileSource) -> str :
    """
        Summarizes the entire document
    """
    mime_type, _ = mimetypes.guess_type(file_source.file_path)
    part = types.Part.from_bytes(data=file_source.content_bytes, mime_type=mime_type)
    content = types.Content(parts=[part], role="user")

    file_summary = await _generate_content(
        model=CHEAP_MODEL,
        messages=[content],
        system_instruction=FILE_SUMMARY_SYSTEM_INSTRUCTION
    )

    return file_summary



async def summarize_chunks(chunks : list[Chunk], file_source: FileSource):
    """
        Performs all the operations required to perform the summary of each chunk content
    """
    try:
        cache_id = str(hash(file_source.file_path))

        cache_name = await _create_cached_file_content(
            model=CHEAP_MODEL,
            system_prompt=CHUNK_SUMMARY_SYSTEM_INSTRUCTION,
            file_path=file_source.file_path,
            cache_id=cache_id
        )    

        for chunk in chunks:
            # generate image description before summarizing
            if chunk.chunk_type == ChunkType.IMAGE:
                image_description = await _generate_image_description(
                    model=CHEAP_MODEL,
                    content=chunk.image_content
                )
                chunk.text_content = image_description

            # create content for ai summary generation
            chunk_content = ""
            if chunk.chunk_type == ChunkType.TABLE:
                chunk_content = chunk.table_content_markdown
            else:
                chunk_content = chunk.text_content
            part = types.Part.from_text(text=chunk_content)
            content = types.Content(parts=[part], role="user")

            summary = await _generate_content(
                model=CHEAP_MODEL,
                messages=[content],
                cache_name=cache_name
            )

            chunk.summary = summary
    except Exception as e:
        logger.error("summarize_chunks : error summarizing chunks")
        raise



async def _generate_image_description(model: str, image_content : Image.Image) -> str:
    """
        Generates the detailed description of the image passed
    """
    buffered = BytesIO()
    image_content.save(buffered, format=image_content.format)
    part = types.Part.from_bytes(data=buffered.getvalue(), mime_type=image_content.get_format_mimetype())
    content = types.Content(parts=[part], role="user")

    image_description = await _generate_content(
        model=model,
        messages=[content],
        system_instruction=IMAGE_DESCRIPTION_SYSTEM_INSTRUCTION
    )

    return image_description



async def generate_search_statement(question: str, summary: str) -> str:
    """
        Generates a better search statement corresponding to the question using the file summary as a context
    """
    messages = []
    # generate question part
    question_part = types.Part.from_text(text="Question : "+question)
    messages.append(types.Content(parts=[question_part], role="user"))

    # generate summary part
    summary_part = types.Part.from_text(text=f"File Summary \n"+summary)
    messages.append(types.Content(parts=[summary_part], role="user"))

    search_statement = await _generate_content(
        model=CHEAP_MODEL,
        messages=messages,
        system_instruction=SEARCH_STATEMENT_SYSTEM_INSTRUCTION
    )

    return search_statement

    

async def get_answer(question: str, chunk_texts: list[str]) -> str:
    messages = []
    # generate question part
    question_part = types.Part.from_text(text="Question : "+question)
    messages.append(types.Content(parts=[question_part], role="user"))

    # generate relevant chunk parts
    relevant_chunk_text = f"Relevant Chunks \n"+"\n".join(chunk_texts)
    relevant_chunk_part = types.Part.from_text(text=relevant_chunk_text)
    messages.append(types.Content(parts=[relevant_chunk_part], role="user"))

    answer = await _generate_content(
        model=CHEAP_MODEL,
        messages=messages,
        system_instruction=DOCUMENT_ANSWERING_SYSTEM_INSTRUCTION
    )

    return answer





    







