from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    """

    # Database Configuration
    DATABASE_URL: str

    # MinIO (S3) Configuration
    S3_ENDPOINT_URL: str
    S3_ACCESS_KEY_ID: str
    S3_SECRET_ACCESS_KEY: str
    S3_BUCKET_NAME: str

    # External API Keys
    OPENROUTER_API_KEY: str = Field(..., min_length=1)

    # Pydantic Settings Configuration
    model_config = SettingsConfigDict(
        env_file=".env",      # Specifies the .env file to load
        env_file_encoding='utf-8',
        extra='ignore'        # Ignore extra fields that might be in the env
    )

# Create a single, globally accessible instance of the settings
settings = Settings()