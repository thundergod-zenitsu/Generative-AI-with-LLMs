from prompts.schemas import TOC_SCHEMA
from config.settings import settings
from core.azure_client import azure_client

class TOCExtractor:

    def extract(self, toc_images):
        messages = [{"role": "system", "content": open("prompts/toc_prompt.txt").read()}]

        # Append each page image
        for (page, img_b64) in toc_images:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "input_image", "image_base64": img_b64},
                    {"type": "text", "text": f"TOC page index {page}"}
                ]
            })

        response = azure_client.vision_chat(settings.MODEL_TOC, messages)
        return response.choices[0].message["content"]
