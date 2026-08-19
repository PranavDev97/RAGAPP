import asyncio
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
from repository.postgresql.chunk import (
    insert_chunks, 
    get_chunks_by_similarity,
    get_chunks_by_keyword,
)



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
    chunks = await asyncio.to_thread(chunk, file_source)

    # Summarize chunks
    await summarize_chunks(chunks=chunks, file_source=file_source)

    # Create chunk summary embeddings
    create_chunk_summary_embeddings(chunks=chunks)

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



async def chat(question: str, doc_id: int) -> tuple[str, list[tuple[int, float]]]:
    file_summary = await get_file_summary(doc_id=doc_id)

    search_statement_info = await generate_search_statement(question=question, summary=file_summary)

    # if the generated search statement has a high confidence score return the statement w/o further retrieval
    if search_statement_info.confidence_score > 0.8:
        return search_statement_info.statement, None

    semantic_chunk_info = await semantic_search(search_statement=search_statement_info.statement, doc_id=doc_id)
    keyword_chunk_info = await get_chunks_by_keyword(doc_id=doc_id, question=question)

    fused_chunk_info = _reciprocal_rank_fusion([semantic_chunk_info, keyword_chunk_info])
    fused_chunk_info = _cutoff_by_elbow(fused_chunk_info)

    chunk_texts = [c[1] for c in fused_chunk_info]
    answer = await get_answer(
        question=question,
        chunk_texts=chunk_texts
    )

    chunk_meta = [(c[0], c[2]) for c in fused_chunk_info]

    return answer, chunk_meta



async def semantic_search(search_statement: str, doc_id: int) -> list[tuple[int, str, float]]:
    hyde = create_search_statement_embedding(search_statement=search_statement)

    chunk_info = await get_chunks_by_similarity(
        doc_id=doc_id,
        vector=hyde,
    )  

    return chunk_info



def _reciprocal_rank_fusion(
    result_lists: list[list[tuple[int, str, float]]],
    k: int = 60,
) -> list[tuple[int, str, float]]:
    """
        Merges multiple ranked (chunk_index, chunk_text, score) lists into a single
        list ordered by Reciprocal Rank Fusion score: each chunk's fused score is
        the sum of 1 / (k + rank) across every list it appears in (rank is
        1-indexed within that list), so chunks ranked highly by multiple
        retrieval strategies are boosted above chunks found by only one.
    """
    fused_scores: dict[int, float] = {}
    chunk_texts: dict[int, str] = {}

    for results in result_lists:
        for rank, (chunk_index, chunk_text, _) in enumerate(results, start=1):
            fused_scores[chunk_index] = fused_scores.get(chunk_index, 0.0) + 1 / (k + rank)
            chunk_texts.setdefault(chunk_index, chunk_text)

    ranked_chunk_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)

    return [(chunk_index, chunk_texts[chunk_index], fused_scores[chunk_index]) for chunk_index in ranked_chunk_indices]



def _cutoff_by_elbow(rows: list[tuple[int, str, float]]) -> list[tuple[int, str, float]]:
    """
        Given rows sorted by score descending, finds the elbow (knee) of the score
        curve using the Kneedle method: scores and ranks are each normalized to
        [0, 1], and the knee is the point of maximum distance from the straight
        line connecting the first and last point. Returns the leading run of rows
        up to and including that point.
    """
    n = len(rows)
    if n <= 2:
        return rows

    scores = [row[2] for row in rows]
    score_range = scores[0] - scores[-1]

    if score_range == 0:
        return rows

    knee_index = 0
    max_distance = -1.0

    for i, score in enumerate(scores):
        x = i / (n - 1)
        y = (score - scores[-1]) / score_range
        distance = abs((1 - x) - y)

        if distance > max_distance:
            max_distance = distance
            knee_index = i

    return rows[:knee_index + 1]