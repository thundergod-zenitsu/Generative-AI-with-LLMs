import os

class Settings:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

    MODEL_TOC = "gpt-4o"
    MODEL_SECTION = "gpt-4o"

    PDF_RENDER_DPI = 200
    MAX_CONCURRENCY = 6
    MAX_RETRIES = 4

settings = Settings()
