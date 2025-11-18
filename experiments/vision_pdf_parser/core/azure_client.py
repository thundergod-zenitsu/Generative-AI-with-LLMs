from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import settings

class AzureOpenAI:
    def __init__(self):
        self.client = OpenAIClient(
            endpoint=settings.AZURE_OPENAI_ENDPOINT,
            credential=AzureKeyCredential(settings.AZURE_OPENAI_API_KEY)
        )

    @retry(
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=8)
    )
    def vision_chat(self, model, messages, max_tokens=4000):
        return self.client.chat_completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_output_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

azure_client = AzureOpenAI()
