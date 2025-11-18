from prompts.schemas import SECTION_SCHEMA
from config.settings import settings
from core.azure_client import azure_client

class SectionExtractor:

    def extract(self, section_title, page_images):
        messages = [
            {
                "role": "system",
                "content": open("prompts/section_prompt.txt").read()
            },
            {
                "role": "user",
                "content": f"SECTION TITLE:\n{section_title}"
            }
        ]

        for img_b64 in page_images:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "input_image", "image_base64": img_b64}
                ]
            })

        response = azure_client.vision_chat(settings.MODEL_SECTION, messages)
        return response.choices[0].message["content"]
