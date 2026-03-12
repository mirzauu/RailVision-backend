"""
Migration script: add file_path and file_url columns to generated_pdfs table.
Run with: PYTHONPATH=. venv/bin/python scripts/migrate_generated_pdfs.py
"""
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres.lvrtwzlgockokaazpfjh:QaZa8dXde7CBJvlb@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"
)

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    conn.execute(text(
        "ALTER TABLE generated_pdfs ADD COLUMN IF NOT EXISTS file_path VARCHAR(1000);"
    ))
    conn.execute(text(
        "ALTER TABLE generated_pdfs ADD COLUMN IF NOT EXISTS file_url VARCHAR(1000);"
    ))
    conn.commit()
    print("✅ Columns file_path and file_url added to generated_pdfs (or already existed).")
