CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS meeting_transcripts (
    chunk_id SERIAL PRIMARY KEY,
    meeting_id VARCHAR(255) NOT NULL,
    speaker_label VARCHAR(50) NOT NULL,
    start_time DOUBLE PRECISION NOT NULL,
    end_time DOUBLE PRECISION NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768),
    CONSTRAINT chk_meeting_transcripts_time_range CHECK (end_time >= start_time),
    CONSTRAINT chk_meeting_transcripts_content_nonempty CHECK (char_length(content) > 0)
);
CREATE INDEX IF NOT EXISTS idx_meeting_transcripts_meeting_id ON meeting_transcripts (meeting_id);
CREATE INDEX IF NOT EXISTS idx_meeting_transcripts_meeting_speaker ON meeting_transcripts (meeting_id, speaker_label);
CREATE INDEX IF NOT EXISTS idx_meeting_transcripts_meeting_start ON meeting_transcripts (meeting_id, start_time);
CREATE INDEX IF NOT EXISTS idx_meeting_transcripts_embedding_hnsw ON meeting_transcripts USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);