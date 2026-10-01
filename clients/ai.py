import httpx
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
from google.genai.errors import ServerError
from models.chunk import Chunk, ChunkType, ChunkMetadata
from models.file import FileSource
from models.ai import SearchStatementAIResponse, ChunkEnrichmentInfoAIResponse


logger = logging.getLogger(__name__)


# constants
CHUNK_SUMMARY_CONCURRENCY = 4
CHEAP_MODEL = "gemini-3.5-flash-lite"


FILE_SUMMARY_SYSTEM_INSTRUCTION = """
You are provided with a full document. Summarize the entire file so as to understand the concepts in the file and also include key entities (such as people, institutions, concepts, or objects) 
present in the file content, and briefly explain their specific role, relevance, and connection to the events or ideas described in the file.
"""
CHUNK_SUMMARY_SYSTEM_INSTRUCTION = """
You are an expert retrieval optimization engine. You will be provided with a full document and an extracted chunk.
Your objective is twofold:
1. Generate a dense contextual summary that situates the chunk within the entire document to maximize dense/keyword vector retrieval.
2. Extract or assign a precise identifying label (name) for the chunk.

Output Instructions
- Summary
   - Function: Situate the chunk in the context of the overall document and detail key entities (people, organizations, technical concepts, datasets, or objects) within the chunk, explaining their direct relationship to the topic.
   - Format: Exactly one continuous paragraph of plain text. 
   - Length: Strict upper bound of 400 tokens.
   - Constraints: Never include conversational filler, introductory preamble ("In this chunk...", "Here is the summary:"), or evaluative commentary.
- Name
    - Identify the explicit identifier present in the chunk, such as a table title, image caption, listing label, or section heading (e.g., "Table 1", "Figure 3.2", "Listing 4: Auth Script", "Appendix B").
    - Crucial Rule: If no explicit name or label is directly stated in the text, you MUST return `null`. Do NOT invent, assume, or infer a descriptive title.
"""
IMAGE_DESCRIPTION_SYSTEM_INSTRUCTION = """

You are given an image. Create a detailed description of the image such that no details of the image is missed. Always include any text inside the image in the decription being generated and explain its usage within the image.
"""
SEARCH_STATEMENT_SYSTEM_INSTRUCTION="""
You are given a user question, a chat history, and a summary of a document. Your task is to rewrite the question as a plain-text declarative statement, using the chat history and the file summary as context so the statement is 
optimized for semantic search retrieval of relevant chunks from that document, and to score how completely that statement already answers the question.

Instructions:

1. Statement generation
   - Rewrite the QUESTION as an affirmative, declarative statement (not a question).
   - Use CHAT HISTORY to resolve pronouns and vague references (it, they, this, that, those,"the same metric", "that table", etc.) into the explicit entity or term they refer to.
   - Use terminology, entities, and phrasing consistent with the FILE SUMARRY and CHAT HISTORY so the statement semantically aligns with how the source document likely discusses the topic.
   - Do not replace a specific term the user used with a more general term from the FILE SUMARRY. Prefer the user's own wording for named entities, metric names, and numbers.
   - If the question names a specific structural element of the document — a table, figure, listing, section, or equation number (e.g., "Table 16", "Section 5.2") — keep that identifier exactly as written in the statement. 
     Do not paraphrase it into a description of what it probably contains.
   - Do not introduce facts, numbers, or claims that are not implied by the QUESTION, CHAT HISTORY or the FILE SUMARRY.

2. Multiple referents and comparisons
   - If the question compares, contrasts, or asks about the relationship between two or more things established earlier in CHAT HISTORY (e.g., "how does that compare to the X you mentioned"), identify each referent separately and 
     include BOTH explicitly and distinctly in the statement. Do not merge them into a single fused claim.
   - Do not invent or assert what the relationship between the two referents is — resolving entities is your job; comparing them is the answering step's job.
   - Bad: "the drop from retrieving more snippets instead of adding more documents" (fuses two separate trends into one nonexistent claim)
   - Good: "comparison of recall increasing with top-k value and recall decreasing with FAISS index size"

3. Confidence score
   Work through these steps in order.

   Step 1 — List what QUESTION needs.
   Identify each distinct fact, figure, or named document element required to fully answer the question.

   Step 2 — Check CHAT HISTORY for each item.
   For every item from Step 1, scan ALL prior turns in CHAT HISTORY (not just the most recent one) for an answer that already states that fact with its specific figure(s). Mark each item FOUND (with the figure) or NOT FOUND. A named 
   table/figure/section counts as FOUND only if a prior turn already quoted its actual content with figures, not just mentioned it.

   Step 3 — Score based on Step 2.
   - If every item is FOUND: confidence_score above 0.8. Compose the statement AS the synthesized answer — state each found figure explicitly, then state the comparison or relationship between them if the question asks for one. Do this 
     even when the items were found in different, separate prior turns.
   - If FILE SUMMARY alone already covers every item, with no CHAT HISTORY needed: score above 0.8, using FILE SUMMARY as the source, composed the same way.
   - If even one item is NOT FOUND in either CHAT HISTORY or FILE SUMMARY: score near 0.1. Compose the statement as a search query for the NOT FOUND item(s) only.

4. Worked example
   CHAT HISTORY includes an earlier turn establishing: "Succinctness scores (1-5, higher is more succinct) for an LLM go from 2.3 at baseline to 3.2 with RAG" — and a later turn establishing: "Fully correct answer rates for the same LLM 
   go from 36 percent at baseline to 60 percent with RAG."
   QUESTION: "Does adding RAG help succinctness as much as it helps correctness?"
   → Step 1: needs (a) the succinctness change with RAG, (b) the correctness change with RAG.
   → Step 2: both FOUND in CHAT HISTORY, from two separate prior turns.
   → confidence_score: 0.9
   → statement: "With RAG, LLM's succinctness score rises from 2.3 to 3.2 (a gain of 0.9 out of 5), while its fully correct answer rate rises from 36% to 60% (a gain of 24 percentage points). RAG produces a larger relative improvement 
     in correctness than in succinctness."
"""
DOCUMENT_ANSWERING_SYSTEM_INSTRUCTION = """
You are given a user question, the search statement used to retrieve it, and the retrieved document chunks. Generate a clear, accurate answer to the question.

Instructions
- Read QUESTION together with CHUNK RETRIEVAL STATEMENT to understand the user's underlying intent, resolved entities, and what specifically is being asked.
- Base your answer strictly on RELEVANT CHUNKS. Do not use outside knowledge and do not infer facts the chunks don't support.
- If the question names a specific table, figure, listing, or section, and a chunk contains that exact element, answer using that chunk's specific content — including its numbers, labels, and values — rather than 
  summarizing related material from elsewhere in the document.
- Preserve exact figures, statistics, and technical terms from the chunks rather than paraphrasing or rounding them. Explain jargon briefly if needed, but don't replace precise language with vaguer language.
- If the question or statement involves comparing two or more things, and the chunks contain data for more than one of them, answer with an explicit comparison using the specific figures for each side — don't just restate one side, 
  and don't treat two distinct topics as a single fused claim to check against the chunks.
- If the chunks contain data for only some of what's being compared, say clearly which part you can answer and which part is missing, rather than declining the whole question.
- If none of the retrieved chunks contain what the question asks for — for example, the  question names a specific table/figure/section that isn't present in the chunks — say so explicitly rather than answering a related but different 
  question. Briefly state what the chunks do cover, if that's useful, but do not present it as if it answers the original question.
- Answer directly and concisely. Don't restate the question or describe your process.
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
        f = await ai_client.aio.files.upload(file=file_path)
        # wait till processing completed
        while f.state.name == "PROCESSING":
            await asyncio.sleep(3)
            f = ai_client.files.get(name=f.name)
    except Exception as e:
        logger.error("_create_cached_file_content : error uploading file",extra={"error": str(e)})
        raise

    # create cache using the uploaded file content            
    try:
        cache = await ai_client.aio.caches.create(
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

    max_retries = 3
    retry_count = 0
    err = None
    while retry_count < max_retries:
        try:
            retry_count += 1
            response = await ai_client.aio.models.generate_content(
                model=model,
                contents=messages,
                config=types.GenerateContentConfig(**config_params)
            )

            return response.text
        # Retry on network errors and server errors (5xx)
        except (ServerError, httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError) as e:
                logger.error("_generate_content :  network/server error ", extra={"error": str(e)})
                err = Exception(f"Network/Server error: {str(e)}")
                await asyncio.sleep(4 ** retry_count)  # Exponential backoff
        except Exception as e:
            logger.error("_generate_content : error generating content", extra={"error": str(e)})
            raise

    if err:
        raise err



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
        response = await ai_client.aio.models.generate_content(
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



async def enrich_chunks(chunks : list[Chunk], file_source: FileSource):
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

        semaphore = asyncio.Semaphore(CHUNK_SUMMARY_CONCURRENCY)
        await asyncio.gather(*(
            _enrich_one_chunk(chunk=chunk, cache_name=cache_name, semaphore=semaphore)
            for chunk in chunks
        ))
    except Exception as e:
        logger.error("enrich_chunks : error summarizing chunks")
        raise



async def _enrich_one_chunk(chunk: Chunk, cache_name: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        # generate image description before summarizing
        if chunk.chunk_type == ChunkType.IMAGE:
            chunk.text_content = await _generate_image_description(
                model=CHEAP_MODEL,
                image_content=chunk.image_content
            )

        # create content for ai chunk enrichment data generation
        chunk_content = ""
        if chunk.chunk_type == ChunkType.TABLE:
            chunk_content = chunk.table_content_markdown
        else:
            chunk_content = chunk.text_content
        part = types.Part.from_text(text=chunk_content)
        content = types.Content(parts=[part], role="user")

        chunk_info: ChunkEnrichmentInfoAIResponse = await _generate_content_structured(
            model=CHEAP_MODEL,
            messages=[content],
            out_schema=ChunkEnrichmentInfoAIResponse,
            cache_name=cache_name,
        )

        chunk.summary = chunk_info.summary
        chunk.metadata = ChunkMetadata(name=chunk_info.name)



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



async def generate_search_statement(question: str, chat_history: str,summary: str) -> SearchStatementAIResponse:
    """
        Generates a better search statement corresponding to the question using the file summary as a context
    """
    messages = []

    # generate chat history part
    chat_history_part = types.Part.from_text(text=f"CHAT HISTORY \n"+chat_history)
    messages.append(types.Content(parts=[chat_history_part], role="user"))

    # generate question part
    question_part = types.Part.from_text(text="QUESTION : "+question)
    messages.append(types.Content(parts=[question_part], role="user"))

    # generate summary part
    summary_part = types.Part.from_text(text=f"FILE SUMMARY \n"+summary)
    messages.append(types.Content(parts=[summary_part], role="user"))

    search_statement_info = await _generate_content_structured(
        model=CHEAP_MODEL,
        messages=messages,
        out_schema=SearchStatementAIResponse,
        system_instruction=SEARCH_STATEMENT_SYSTEM_INSTRUCTION
    )

    return search_statement_info

    

async def get_answer(question: str, retrieval_statement: str, chunk_texts: list[str]) -> str:
    messages = []
    # generate question part
    question_part = types.Part.from_text(text="QUESTION : "+question)
    messages.append(types.Content(parts=[question_part], role="user"))

    # generate retrieval statement part
    statement_part = types.Part.from_text(text="CHUNK RETRIEVAL STATEMENT : "+retrieval_statement)
    messages.append(types.Content(parts=[statement_part], role="user"))

    # generate relevant chunk parts
    relevant_chunk_text = f"RELEVANT CHUNKS \n"+"\n".join(chunk_texts)
    relevant_chunk_part = types.Part.from_text(text=relevant_chunk_text)
    messages.append(types.Content(parts=[relevant_chunk_part], role="user"))

    answer = await _generate_content(
        model=CHEAP_MODEL,
        messages=messages,
        system_instruction=DOCUMENT_ANSWERING_SYSTEM_INSTRUCTION
    )

    return answer





    







