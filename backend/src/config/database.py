from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from src.config.settings import settings

db_url = settings.database_url or ""
if "${SUPABASE_DB_PASSWORD}" in db_url:
    db_url = db_url.replace("${SUPABASE_DB_PASSWORD}", settings.supabase_db_password or "")
elif "postgres:@" in db_url and settings.supabase_db_password:
    db_url = db_url.replace("postgres:@", f"postgres:{settings.supabase_db_password}@")
if db_url.startswith("sqlite"):
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
else:
    if "postgresql+asyncpg" in db_url:
        db_url = db_url.replace("postgresql+asyncpg", "postgresql+psycopg2")
    if "supabase.co" in db_url and "sslmode=" not in db_url:
        db_url = db_url + ("&sslmode=require" if "?" in db_url else "?sslmode=require")
    engine = create_engine(db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

if settings.auto_create_tables:
    from src.infrastructure.database import models  # noqa: F401
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
