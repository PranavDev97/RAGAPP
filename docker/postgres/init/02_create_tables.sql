CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL
);

CREATE TABLE document_source (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL, 
    content_summary TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL
);

CREATE TABLE document_chunk (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL REFERENCES document_source(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_type TEXT NOT NULL,
    chunk_summary TEXT NOT NULL,
    chunk_embedding VECTOR(768),
    chunk_tsvector TSVECTOR
        GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,

    UNIQUE (doc_id, chunk_index)
);

CREATE TABLE user_document (
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    doc_id INT REFERENCES document_source(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, doc_id)
);

CREATE INDEX idx_document_chunk_embedding ON document_chunk USING hnsw(chunk_embedding vector_cosine_ops);
CREATE INDEX idx_document_chunk_tsvector ON document_chunk USING GIN (chunk_tsvector);