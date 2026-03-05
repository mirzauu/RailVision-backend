-- Migration: Create generated_spreadsheets table
-- Run this SQL on your Supabase PostgreSQL instance before using the spreadsheet tool.

CREATE TABLE IF NOT EXISTS generated_spreadsheets (
    id               VARCHAR      PRIMARY KEY DEFAULT gen_random_uuid()::text,
    conversation_id  VARCHAR      NOT NULL REFERENCES conversations(id)  ON DELETE CASCADE,
    org_id           VARCHAR      NOT NULL REFERENCES organizations(id)  ON DELETE CASCADE,
    title            VARCHAR(500) NOT NULL,
    file_path        VARCHAR(1000),  -- relative server path, e.g. storage/spreadsheets/<uuid>.xlsx
    file_url         VARCHAR(1000),  -- public download URL
    sheet_count      INTEGER      DEFAULT 1,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    deleted_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_generated_spreadsheets_conversation_id
    ON generated_spreadsheets (conversation_id);

CREATE INDEX IF NOT EXISTS ix_generated_spreadsheets_org_id
    ON generated_spreadsheets (org_id);
