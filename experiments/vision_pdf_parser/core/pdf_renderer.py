import fitz  # PyMuPDF
from PIL import Image
import io
import base64

from config.settings import settings

class PDFRenderer:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)

    def num_pages(self):
        return len(self.doc)

    def page_to_png_base64(self, page_number):
        page = self.doc.load_page(page_number)
        zoom = settings.PDF_RENDER_DPI / 72
        mat = fitz.Matrix(zoom, zoom)

        pix = page.get_pixmap(matrix=mat, alpha=False)

        img_bytes = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).tobytes()

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        output = io.BytesIO()
        img.save(output, format="PNG", optimize=True)
        encoded = base64.b64encode(output.getvalue()).decode()

        return encoded

    def render_pages(self, start, end):
        for i in range(start, end + 1):
            yield i, self.page_to_png_base64(i)

