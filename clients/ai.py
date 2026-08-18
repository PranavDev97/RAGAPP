import asyncio
import logging
import mimetypes
from pathlib import Path
from typing import Optional
from io import BytesIO
from typing import Type
from PIL import Image
from pydantic import BaseModel
from google import genai
from google.genai import types
from models.chunk import Chunk, ChunkType
from models.file import FileSource
from models.ai import SearchStatementAIResponse


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

Limit your response to 400 tokens. Output only this contextual summary and entity explanations as single paragraph in plain text — do not include any additional commentary or introductory text.
"""
IMAGE_DESCRIPTION_SYSTEM_INSTRUCTION = """
You are given an image. Create a detailed description of the image such that no details of the image is missed. Always include any text inside the image in the decription being generated and explain its usage within the image.
"""
SEARCH_STATEMENT_SYSTEM_INSTRUCTION="""
You are given a user question and a summary of a document. Your task is to rewrite the question as a plain-text declarative statement, using the file summary as context so the 
statement is optimized for semantic search retrieval of relevant chunks from that document.

Instructions:

1. Statement generation
   - Rewrite the question as an affirmative, declarative statement (not a question).
   - Use terminology, entities, and phrasing consistent with the file summary so the 
     statement semantically aligns with how the source document likely discusses the topic.
   - Do not introduce facts, numbers, or claims that are not implied by the question or 
     the file summary.

2. Confidence score
   - Assign a confidence_score between 0.1 and 1.0 that reflects how complete an answer 
     the statement itself is:
     - If the statement is useful only as a search query (i.e., it does not itself answer 
       the question and further chunk retrieval is required), assign a score close to 0.1.
     - If the statement is a complete, accurate, and self-contained answer to the question 
       based solely on the file summary, assign a score close to 1.0.
   - Only assign a score above 0.8 if the statement could be used directly as the final 
     answer to the user's question, with no further chunk retrieval needed.
   - Do not assign a score above 0.8 if any additional information from the document 
     would be needed to fully answer the question.
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



async def _generate_content_structured(model: str, messages: list[types.Content],
                                        out_schema:  Type[BaseModel], system_instruction: Optional[str]=None, 
                                        cache_name: Optional[str]=None
) -> Type[BaseModel]:
    """
        Generates AI content as the given out_schema object based on the input messages given
    """
    config_params = {
        "temperature": 0.2,
        "max_output_tokens": 10000,
        "response_mime_type": "application/json",
        "response_schema": out_schema,
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

        return response.parsed
    except Exception as e:
        logger.error("_generate_content_structured : error generating content", extra={"error": str(e)})
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
                    image_content=chunk.image_content
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



async def generate_search_statement(question: str, summary: str) -> SearchStatementAIResponse:
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

    search_statement_info = await _generate_content_structured(
        model=CHEAP_MODEL,
        messages=messages,
        out_schema=SearchStatementAIResponse,
        system_instruction=SEARCH_STATEMENT_SYSTEM_INSTRUCTION
    )

    return search_statement_info

    

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





    







