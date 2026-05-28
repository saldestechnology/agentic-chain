from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    debug: bool = False
    env: Literal['production', 'development'] = 'development'
    log_level: Optional[Literal['DEBUG', 'INFO', 'ERROR']]
    