import os
from pydantic import BaseModel, Field
from typing import List

class Settings(BaseModel):
    env: str = Field(default=os.getenv("SMRITI_ENV", "development"))
    debug: bool = Field(default=os.getenv("SMRITI_DEBUG", "true").lower() == "true")
    database_url: str = Field(default=os.getenv("SMRITI_DATABASE_URL", "sqlite:///./smriti.db"))
    storage_dir: str = Field(default=os.getenv("SMRITI_STORAGE_DIR", "./storage"))
    secret_key: str = Field(default=os.getenv("SMRITI_SECRET_KEY", "dev-secret-key-smriti-memory-layer"))
    cors_origins: List[str] = Field(default=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ])
    embedding_provider: str = Field(default=os.getenv("SMRITI_EMBEDDING_PROVIDER", "local"))
    embedding_model: str = Field(default=os.getenv("SMRITI_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

settings = Settings()
