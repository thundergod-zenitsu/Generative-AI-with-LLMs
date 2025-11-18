import asyncio
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from core.section_extractor import SectionExtractor

section_extractor = SectionExtractor()

async def process_section(renderer, title, page_start, page_end):
    # heavy work moved to thread pool
    loop = asyncio.get_event_loop()

    # get image pages
    page_images = await loop.run_in_executor(
        None,
        lambda: [img for _, img in renderer.render_pages(page_start - 1, page_end - 1)]
    )

    # GPT extraction
    result = await loop.run_in_executor(
        None,
        lambda: section_extractor.extract(title, page_images)
    )

    return (page_start, page_end, title, result)


async def run_all_sections(renderer, toc_struct):
    tasks = []

    def schedule(nodes):
        for n in nodes:
            tasks.append(process_section(
                renderer,
                n["title"],
                n["page_start"],
                n["page_end"]
            ))
            if n["children"]:
                schedule(n["children"])

    schedule(toc_struct["sections"])

    results = await asyncio.gather(*tasks)
    return { (a,b,c):d for (a,b,c,d) in results }
