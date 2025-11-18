import json
from core.pdf_renderer import PDFRenderer
from core.toc_extractor import TOCExtractor
from core.json_builder import JSONBuilder
from core.parallel_engine import run_all_sections
import asyncio

def run_pipeline(pdf_path, toc_pages):
    print("📄 Rendering PDF...")
    renderer = PDFRenderer(pdf_path)

    toc_imgs = []
    for p in range(toc_pages):
        toc_imgs.append((p+1, renderer.page_to_png_base64(p)))

    print("🔍 Extracting TOC...")
    toc_extractor = TOCExtractor()
    toc_raw = toc_extractor.extract(toc_imgs)
    toc_json = json.loads(toc_raw)

    print("📐 Assigning page ranges...")
    builder = JSONBuilder()
    toc_struct = builder.assign_page_ranges(toc_json, renderer.num_pages())

    print("⚡ Extracting sections in parallel...")
    content_map = asyncio.run(run_all_sections(renderer, toc_struct))

    print("🌳 Building hierarchical JSON...")
    final_json = builder.integrate_content(toc_struct, content_map)

    return final_json


if __name__ == "__main__":
    pdf = "sample.pdf"
    toc_pages = 3
    final_output = run_pipeline(pdf, toc_pages)

    with open("structured_output.json", "w") as f:
        json.dump(final_output, f, indent=2)

    print("🎉 Done! Output written to structured_output.json")
