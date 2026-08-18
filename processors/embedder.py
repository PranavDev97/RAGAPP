import numpy
from sentence_transformers import SentenceTransformer
from models.chunk import Chunk


# constants
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v2-moe"
EMBEDDING_DIMENSION = 768
PROMPT_NAME_PASSAGE = "passage"
PROMPT_NAME_QUERY = "query"



embed_model : SentenceTransformer | None = None
async def create_embed_model():
    global embed_model
    embed_model = SentenceTransformer(
        EMBED_MODEL_NAME, truncate_dim = EMBEDDING_DIMENSION, trust_remote_code=True
    )



def _create_embedding(content: list[str], prompt_name: str) -> list[list[float]]:
    embeddings = embed_model.encode(
        content,
        prompt_name=prompt_name
    )
    return embeddings.tolist() 



def create_chunk_summary_embeddings(chunks : list[Chunk]):
    summaries = [chunk.summary for chunk in chunks]
    embeddings = _create_embedding(content=summaries, prompt_name=PROMPT_NAME_PASSAGE)
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding



def create_question_embedding(question: str) -> list[float]:
    embeddings = _create_embedding(content=[question], prompt_name=PROMPT_NAME_QUERY)
    return embeddings[0]



def create_search_statement_embedding(search_statement: str) -> list[float]:
    embeddings = _create_embedding(content=[search_statement], prompt_name=PROMPT_NAME_PASSAGE)
    return embeddings[0]