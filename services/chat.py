from models.file import FileSource
from models.chunk import Chunk
from pathlib import Path
from processors.chunker import chunk
from processors.embedder import (
    create_chunk_summary_embeddings, 
    create_search_statement_embedding,
)
from clients.ai import (
    summarize_file, 
    summarize_chunks, 
    get_answer,
    generate_search_statement,
)
from repository.postgresql.file import insert_document_source, get_file_summary
from repository.postgresql.chunk import insert_chunks, get_chunks_by_similarity



UPLOAD_DIR = Path("files/chat")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)



async def upload_and_process(file_source: FileSource) -> int:
    # Save file
    file_path = UPLOAD_DIR / file_source.name
    file_source.file_path = file_path

    with file_path.open("wb") as buffer:
        buffer.write(file_source.content_bytes)

    # Create file summary
    file_source.content_summary = await summarize_file(file_source=file_source)

    # Create chunks
    chunks = await chunk(file_source=file_source)

    # Summarize chunks
    await summarize_chunks(chunks=chunks, file_source=file_source)

    # Create chunk summary embeddings
    await create_chunk_summary_embeddings(chunks=chunks)

    doc_id = await _insert_processed_file_data_to_db(file_source=file_source, chunks=chunks)

    return doc_id



async def _insert_processed_file_data_to_db(file_source: FileSource, chunks: list[Chunk]) -> int:
    doc_id = await insert_document_source(
        name=file_source.name, 
        file_path=file_source.file_path,
        summary=file_source.content_summary
    )
    await insert_chunks(doc_id=doc_id, chunks=chunks)

    return doc_id



async def chat(question: str, doc_id: int) -> tuple[str, list[int]]:
    hyde = await _generate_hyde(question=question, doc_id=doc_id)

    chunk_info = await get_chunks_by_similarity(
        doc_id=doc_id,
        vector=hyde,
    )

    chunk_texts = [c[1] for c in chunk_info ]

    if not chunk_texts:
        return None, None

    answer = await get_answer(
        question=question,
        chunk_texts=chunk_texts
    )

    chunk_meta = [(c[0],c[2]) for c in chunk_info]

    return answer, chunk_meta



async def _generate_hyde(question: str, doc_id: int) -> list[float]:
    """
        Generate Hypothetical Document Embedding for better semantic search
    """
    file_summary = await get_file_summary(doc_id=doc_id)

    search_statement = await generate_search_statement(question=question, summary=file_summary)

    print(f"search statement : {search_statement}")

    hyde = await create_search_statement_embedding(search_statement=search_statement)

    return hyde






    