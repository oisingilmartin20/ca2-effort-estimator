CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE ticket_embeddings (
  id              SERIAL PRIMARY KEY,
  issue_key       VARCHAR(128) NOT NULL UNIQUE,
  title           TEXT,
  description     TEXT NOT NULL,
  story_points    INTEGER NOT NULL,
  embedding       vector(384),
  embedding_model VARCHAR(128) NOT NULL,
  created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ticket_embeddings_hnsw_idx
  ON ticket_embeddings
  USING hnsw (embedding vector_cosine_ops);
