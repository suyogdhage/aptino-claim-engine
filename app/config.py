from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    POLICY_PDF_PATH: str = "./data/policy.pdf"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    API_BASE_URL: str = ""
    FORCE_MOCK_LLM: bool = False
    CORS_ORIGINS: str = "http://localhost:8501"

    class Config:
        env_file = ".env"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()
