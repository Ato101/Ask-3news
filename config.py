import os
from pydantic_settings import BaseSettings


class Setting(BaseSettings):
    qdrant_url: str
    qdrant_api_key: str
    openai_api_key: str

    class Config:
        env_file = '.env'


settings = Setting()
