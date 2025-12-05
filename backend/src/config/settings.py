from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")
    # Application Environment
    app_env: str = "development"  # development or production
    
    # Database
    database_url: Optional[str] = "sqlite:///./app.db"
    
    # Security
    jwt_secret: Optional[str] = "dev-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # LLM Providers
    openai_api_key: Optional[str] = None
    claude_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    
    # Vector Database
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "ai-c-suite"
    
    # Logging
    log_level: str = "INFO"
    
    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    
    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"
    
settings = Settings()

