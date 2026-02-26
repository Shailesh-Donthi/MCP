"""Core runtime settings for local MCP execution."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "personnel_reporting"

    ALLOWED_ORIGINS: str = "*"
    MCP_HOST: str = "127.0.0.1"
    MCP_PORT: int = 8090
    MCP_ALLOW_ANONYMOUS_ACCESS: bool = False
    MCP_ANON_SCOPE_LEVEL: str = "state"
    MCP_ENABLE_DB_TEST_ENDPOINT: bool = False
    MCP_ENABLE_DEMO_LOGIN: bool = False
    MCP_DEMO_USERNAME: str = "9999"
    MCP_DEMO_PIN: str = "9999"
    MCP_DEMO_LOGIN_TOKEN_TTL_MINUTES: int = 480

    REDIS_URL: str | None = None

    MCP_LOG_LEVEL: str = "INFO"
    MCP_LOG_FILE: str | None = "logs/mcp_app.log"
    MCP_LOG_MAX_BYTES: int = 10485760
    MCP_LOG_BACKUP_COUNT: int = 5

    JWT_SECRET_KEY: str = "change-me"
    JWT_ALGORITHM: str = "HS256"


settings = Settings()
