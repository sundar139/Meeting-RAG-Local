ALTER TABLE meeting_transcripts
ADD COLUMN IF NOT EXISTS chunk_key VARCHAR(80);
CREATE UNIQUE INDEX IF NOT EXISTS uq_meeting_transcripts_meeting_chunk_key ON meeting_transcripts (meeting_id, chunk_key);