"""
PageIndexTreeGenerator (Azure + PDF Edition)

This script replicates the PageIndex backend's end-to-end generation logic,
taking a raw PDF as input. It simulates the two-stage PageIndex process:

1.  (Pre-Step) Extracts raw text from the PDF using PyMuPDF.
2.  (Stage 1 Sim) Uses an LLM to discover the document's global
    hierarchical structure from the flat text.
3.  (Stage 2 Sim) Uses a high-performance parallel recursive descent
    architecture to build the final, lightweight JSON tree.[4, 1]
"""

import os
import json
import logging
import threading
import time
from openai import AzureOpenAI
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Attempt to import PyMuPDF (fitz)
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF not found.")
    print("Please install it using: pip install PyMuPDF")
    exit(1)

# --- Configuration & Setup ---

# Configure logging for clear, multi-threaded output
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] (%(threadName)-10s) %(message)s",
)

# --- Step 1: Define the PageIndex Node Schema ---

@dataclass
class PageIndexNode:
    """
    Represents a single node in the PageIndex tree, matching the
    documented JSON structure.[5, 1]

    This is the "lightweight JSON object"  that is optimized
    for an LLM to navigate.[5]
    """
    title: str
    node_id: str
    page_index: int  # The page number where this node's content begins [1]
    summary: str     # An LLM-generated summary for navigation 
    nodes: List['PageIndexNode'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Recursively serializes the node and its children to a
        JSON-compatible dictionary.
        """
        return {
            "title": self.title,
            "node_id": self.node_id,
            "page_index": self.page_index,
            "summary": self.summary,
            "nodes": [child.to_dict() for child in self.nodes]
        }


# --- Step 2: Define the High-Performance Tree Generator ---

class PageIndexTreeGenerator:
    """
    Simulates the PageIndex backend's two-stage generation process 
    using a parallelized, recursive architecture with an AzureOpenAI client.
    """

    def __init__(self, azure_client: AzureOpenAI, deployment_name: str, max_workers: int = 10):
        """
        Initializes the generator.

        Args:
            azure_client: An initialized AzureOpenAI client instance.
            deployment_name: The name of your Azure deployment (model name).
            max_workers: The number of parallel threads for processing
                         document sections.
        """
        self.client = azure_client
        self.deployment_name = deployment_name
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, 
            thread_name_prefix="PageIndex_Worker"
        )
        self._node_counter = 0
        self._counter_lock = threading.Lock()
        logging.info(
            f"PageIndexTreeGenerator initialized with {max_workers} workers."
        )

    def _generate_node_id(self) -> str:
        """
        Generates a thread-safe, sequential, unique node ID.[1]
        """
        with self._counter_lock:
            self._node_counter += 1
            return f"{self._node_counter:04d}"

    def _generate_summary(self, content_chunk: str, title: str) -> str:
        """
        *** Simulates a key part of Stage 2 Generation  ***
        
        Uses an LLM to generate a concise summary for a node. This summary
        is critical for the retrieval LLM to make navigation
        decisions.[6]
        """
        if not content_chunk.strip():
            return "This section has no content."

        # Truncate for efficiency if content is enormous
        if len(content_chunk) > 10000:
            content_chunk = content_chunk[:10000] + "..."

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                temperature=0.0,
                messages=
            )
            summary = response.choices.message.content
            return summary.strip()
        except Exception as e:
            logging.error(f"LLM summary generation failed for '{title}': {e}")
            return (content_chunk[:200] + '...') # Fallback
            
    def _discover_sections_at_level(
        self, 
        parent_content: str, 
        current_level: int
    ) -> (str, List]):
        """
        *** This simulates Stage 1: The "PageIndex OCR"  ***
        
        This is the core AI-driven discovery function. It uses an LLM to scan
        a block of text and find all *immediate* child sections at a
        specific heading level.
        """
        heading_char = "#" * current_level
        
        system_prompt = f"""
        You are a document structure parser, simulating 'PageIndex OCR'.
        Your task is to analyze the provided text and extract all sections 
        that start with a '{heading_char}' heading (level {current_level}).

        You MUST return a JSON object with two keys:
        1. "parent_content": A string containing ALL text from the
           beginning of the document *until* the *first*
           '{heading_char}' heading. This is the parent's own text.
        2. "child_sections": A list of JSON objects, one for each
           section you find at this level. Each object must have:
           - "title": The string title of the section (e.g., "1.1 Introduction").
           - "page_index": A placeholder integer. Use {self._node_counter + 1}.
           - "full_content": The *full* text content *under* that heading,
             including all subsections, up until the *next*
             '{heading_char}' heading or the end of the text.

        If no '{heading_char}' headings are found, return an empty
        "child_sections" list and the entire text as "parent_content".
        Your response MUST be *only* the JSON object.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": parent_content}
                ]
            )
            
            result = json.loads(response.choices.message.content)
            
            if "parent_content" not in result or "child_sections" not in result:
                raise ValueError("LLM returned malformed JSON structure.")
            if not isinstance(result["child_sections"], list):
                 raise ValueError("LLM returned 'child_sections' that was not a list.")

            return result["parent_content"], result["child_sections"]

        except Exception as e:
            logging.error(f"LLM structure discovery failed at level {current_level}: {e}")
            return parent_content, # Fallback

    def _process_node_job(self, job_data: tuple) -> PageIndexNode:
        """
        *** This is the parallel worker function (simulates Stage 2)  ***
        
        This function runs in a separate thread for *each* node. It:
        1. Discovers the structure *within* its assigned content (Stage 1 Sim).
        2. Generates the summary for its *own* content (Stage 2 Sim).
        3. Recursively spawns parallel jobs for all its children.
        """
        title, page_index, full_content, next_level = job_data
        
        try:
            this_node_content, child_sections_data = self._discover_sections_at_level(
                parent_content=full_content,
                current_level=next_level
            )
        except Exception as e:
            logging.error(f"Failed parsing children for '{title}': {e}")
            this_node_content = full_content
            child_sections_data =

        logging.info(f"Generating summary for: {title}")
        summary = self._generate_summary(this_node_content, title)
        
        node = PageIndexNode(
            title=title,
            node_id=self._generate_node_id(),
            page_index=page_index,
            summary=summary
        )

        if child_sections_data:
            child_jobs =
            for child_data in child_sections_data:
                if not all(k in child_data for k in ('title', 'page_index', 'full_content')):
                    logging.warning(f"Skipping malformed child data under '{title}'")
                    continue
                
                child_jobs.append((
                    child_data['title'],
                    child_data['page_index'],
                    child_data['full_content'],
                    next_level + 1
                ))
            
            try:
                node.nodes = list(self.executor.map(self._process_node_job, child_jobs))
            except Exception as e:
                 logging.error(f"Error processing sub-jobs for {title}: {e}")

        logging.info(f"Finished processing node: {title}")
        return node

    def _extract_raw_text_from_pdf(self, pdf_file_path: str) -> str:
        """
        (Pre-Step) Extracts full, raw text from a PDF file.
        """
        logging.info(f"Starting PDF text extraction for: {pdf_file_path}")
        full_text = ""
        try:
            with fitz.open(pdf_file_path) as doc:
                for page in doc:
                    full_text += page.get_text() + "\n"
            logging.info(f"Successfully extracted {len(full_text)} characters.")
            return full_text
        except Exception as e:
            logging.critical(f"Failed to read or parse PDF file: {e}")
            raise

    def _generate_tree_from_text(self, raw_document_text: str) -> List]:
        """
        Internal method that orchestrates the parallel generation
        from a raw text block.
        """
        start_time = time.perf_counter()
        
        with self._counter_lock:
            self._node_counter = 0
        
        logging.info("Discovering top-level document structure (Stage 1 Sim)...")
        try:
            _root_content, top_level_sections = self._discover_sections_at_level(
                parent_content=raw_document_text,
                current_level=1
            )
        except Exception as e:
            logging.critical(f"Failed to discover root structure: {e}")
            return

        if not top_level_sections:
            logging.warning("No top-level sections found. Check document format.")
            return

        jobs =
        for section in top_level_sections:
            jobs.append((
                section['title'],
                section['page_index'],
                section['full_content'],
                2
            ))

        logging.info(f"Submitting {len(jobs)} top-level sections to thread pool...")
        
        root_nodes: List[PageIndexNode] =
        try:
            root_nodes = list(self.executor.map(self._process_node_job, jobs))
        except Exception as e:
            logging.critical(f"A critical error occurred during parallel processing: {e}")

        end_time = time.perf_counter()
        logging.info("--- Tree Generation Complete ---")
        logging.info(f"Total time: {end_time - start_time:.2f} seconds")
        logging.info(f"Total nodes generated: {self._node_counter}")
        
        return [node.to_dict() for node in root_nodes]

    def generate_tree_from_pdf(self, pdf_file_path: str) -> List]:
        """
        The main public method that orchestrates the full, end-to-end
        generation process from a raw PDF file.
        
        Args:
            pdf_file_path: The local path to the PDF document.

        Returns:
            A list of dictionaries representing the root-level nodes
            of the generated PageIndex tree.
        """
        logging.info("--- Starting PageIndex PDF Generation Pipeline ---")
        
        # --- PRE-STEP: PDF-to-Text ---
        raw_text = self._extract_raw_text_from_pdf(pdf_file_path)
        
        if not raw_text:
            logging.error("PDF extraction resulted in empty text. Aborting.")
            return

        logging.warning("PDF extraction complete. Simulating PageIndex OCR...")
        logging.warning(
            "Note: This simulation uses an LLM on flat text. "
            "The real PageIndex OCR uses a vision-language model "
            "to read the PDF's visual structure."
        )
        
        # --- STAGES 1 & 2: Generate Tree from Text ---
        return self._generate_tree_from_text(raw_text)







# --- Example Usage (place at the end of the generate_tree.py file) ---

if __name__ == "__main__":
    
    # --- AzureOpenAI Client Setup ---
    try:
        azure_client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("OPENAI_API_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        )
        
        # *** IMPORTANT ***
        # Set this to the name of your deployed model in Azure AI Studio
        DEPLOYMENT_NAME = "gpt-4-turbo" # <-- CHANGE THIS
        
        # *** IMPORTANT ***
        # Set this to the path of your input PDF
        PDF_FILE_PATH = "./my_large_report.pdf" # <-- CHANGE THIS
        
        if not os.path.exists(PDF_FILE_PATH):
            logging.critical(f"Input file not found: {PDF_FILE_PATH}")
            # Create a dummy file for testing if it doesn't exist
            try:
                # This is a simple text, not a real PDF.
                # For a real test, please use an actual PDF file.
                with open("my_large_report.pdf", "w") as f:
                    f.write("# Ch 1\nText 1\n## Sec 1.1\nText 1.1\n# Ch 2\nText 2")
                logging.warning("Dummy file created. For a real test, use a real PDF.")
                PDF_FILE_PATH = "my_large_report.pdf" # This won't be read by fitz as a PDF
            except Exception:
                pass # Will fail below, which is expected
                
        logging.info(f"AzureOpenAI client initialized. Target deployment: {DEPLOYMENT_NAME}")
        
    except Exception as e:
        logging.critical(f"Failed to initialize AzureOpenAI client: {e}")
        logging.critical(
            "Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
            "and OPENAI_API_VERSION environment variables."
        )
        exit(1)

    
    # Instantiate the generator
    generator = PageIndexTreeGenerator(
        azure_client=azure_client,
        deployment_name=DEPLOYMENT_NAME,
        max_workers=10  # Use 10 parallel workers
    )
    
    # Run the full, end-to-end PDF-to-Tree process
    final_pageindex_tree = generator.generate_tree_from_pdf(PDF_FILE_PATH)
    
    if final_pageindex_tree:
        print("\n\n--- FINAL PAGEINDEX TREE (JSON Output) ---")
        
        # Pretty-print the final JSON
        print(json.dumps(final_pageindex_tree, indent=2))
        
        # Save to a file
        output_filename = f"{os.path.basename(PDF_FILE_PATH)}.json"
        with open(output_filename, "w") as f:
            json.dump(final_pageindex_tree, f, indent=2)
        logging.info(f"Final tree saved to {output_filename}")
    else:
        logging.error("Tree generation failed.")