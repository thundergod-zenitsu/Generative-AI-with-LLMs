"""
Production PageIndex Tree Generator with Azure OpenAI
Features:
- Azure OpenAI integration with rate limiting
- Token-based chunking with tiktoken
- Layout detection (bounding boxes) from PDF
- Disk-based intermediate storage for memory efficiency
- Resumable processing with checkpoints
- Comprehensive error handling and logging
"""

import json
import asyncio
import time
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
import hashlib
import pickle
from collections import deque

# Azure OpenAI
from openai import AsyncAzureOpenAI
from openai import RateLimitError, APIError

# PDF Processing
import fitz  # PyMuPDF
import tiktoken

# Rate limiting
from asyncio import Semaphore
import aiofiles


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pageindex.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box for layout elements"""
    x0: float
    y0: float
    x1: float
    y1: float
    page_num: int
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class LayoutElement:
    """Text element with layout information"""
    text: str
    bbox: BoundingBox
    font_size: float
    font_name: str
    is_bold: bool = False
    is_italic: bool = False
    
    @property
    def is_likely_title(self) -> bool:
        """Heuristic to detect if this is likely a title/heading"""
        # Larger font size, short text, possibly bold
        return (
            self.font_size > 12 and
            len(self.text) < 200 and
            (self.is_bold or self.text.isupper() or self.text.istitle())
        )
    
    @property
    def hierarchy_level(self) -> int:
        """Estimate hierarchy level based on font size"""
        if self.font_size >= 20:
            return 0  # Chapter/main title
        elif self.font_size >= 16:
            return 1  # Section
        elif self.font_size >= 14:
            return 2  # Subsection
        else:
            return 3  # Subsubsection or paragraph


@dataclass
class DocumentNode:
    """Node in the PageIndex tree with layout metadata"""
    title: str
    node_id: str
    start_page: int
    end_page: int
    text: str = ""
    summary: str = ""
    level: int = 0
    parent_id: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    token_count: int = 0
    children: List['DocumentNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_children: bool = True) -> Dict[str, Any]:
        """Convert to PageIndex-compatible format"""
        result = {
            "title": self.title,
            "node_id": self.node_id,
            "start_index": self.start_page,
            "end_index": self.end_page,
        }
        
        if self.summary:
            result["summary"] = self.summary
        
        if self.bbox:
            result["bbox"] = {
                "x0": self.bbox.x0,
                "y0": self.bbox.y0,
                "x1": self.bbox.x1,
                "y1": self.bbox.y1,
                "page": self.bbox.page_num
            }
        
        if self.token_count:
            result["token_count"] = self.token_count
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        if include_children and self.children:
            result["nodes"] = [child.to_dict() for child in self.children]
        
        return result


class TokenCounter:
    """Token counting with tiktoken"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])


class RateLimiter:
    """Rate limiter for Azure OpenAI API calls"""
    
    def __init__(
        self,
        max_requests_per_minute: int = 60,
        max_tokens_per_minute: int = 150000
    ):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        
        self.request_times = deque()
        self.token_usage = deque()
        self.lock = asyncio.Lock()
        
    async def acquire(self, estimated_tokens: int = 1000):
        """Acquire permission to make API call"""
        async with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            while self.request_times and now - self.request_times[0] > 60:
                self.request_times.popleft()
            
            while self.token_usage and now - self.token_usage[0][0] > 60:
                self.token_usage.popleft()
            
            # Check rate limits
            current_requests = len(self.request_times)
            current_tokens = sum(t[1] for t in self.token_usage)
            
            # Wait if necessary
            while (
                current_requests >= self.max_requests_per_minute or
                current_tokens + estimated_tokens > self.max_tokens_per_minute
            ):
                wait_time = 60 - (now - self.request_times[0]) if self.request_times else 1
                logger.info(f"Rate limit reached. Waiting {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
                
                # Recalculate
                now = time.time()
                while self.request_times and now - self.request_times[0] > 60:
                    self.request_times.popleft()
                while self.token_usage and now - self.token_usage[0][0] > 60:
                    self.token_usage.popleft()
                
                current_requests = len(self.request_times)
                current_tokens = sum(t[1] for t in self.token_usage)
            
            # Record this request
            self.request_times.append(now)
            self.token_usage.append((now, estimated_tokens))


class PDFLayoutExtractor:
    """Extract text with layout information from PDF"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.total_pages = len(self.doc)
        
    def extract_page_layout(self, page_num: int) -> List[LayoutElement]:
        """Extract layout elements from a page"""
        page = self.doc[page_num]
        elements = []
        
        # Extract text blocks with formatting
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if not text:
                            continue
                        
                        bbox = BoundingBox(
                            x0=span["bbox"][0],
                            y0=span["bbox"][1],
                            x1=span["bbox"][2],
                            y1=span["bbox"][3],
                            page_num=page_num
                        )
                        
                        # Extract font properties
                        font_name = span.get("font", "")
                        font_size = span.get("size", 12)
                        is_bold = "bold" in font_name.lower()
                        is_italic = "italic" in font_name.lower()
                        
                        element = LayoutElement(
                            text=text,
                            bbox=bbox,
                            font_size=font_size,
                            font_name=font_name,
                            is_bold=is_bold,
                            is_italic=is_italic
                        )
                        
                        elements.append(element)
        
        return elements
    
    def extract_page_text(self, page_num: int) -> str:
        """Extract plain text from a page"""
        page = self.doc[page_num]
        return page.get_text()
    
    def extract_titles_from_range(
        self,
        start_page: int,
        end_page: int
    ) -> List[Tuple[str, BoundingBox, int]]:
        """Extract likely titles from page range"""
        titles = []
        
        for page_num in range(start_page, end_page + 1):
            if page_num >= self.total_pages:
                break
            
            elements = self.extract_page_layout(page_num)
            
            # Filter for title-like elements
            for elem in elements:
                if elem.is_likely_title:
                    titles.append((elem.text, elem.bbox, elem.hierarchy_level))
        
        return titles
    
    def close(self):
        """Close PDF document"""
        self.doc.close()


class AzureOpenAIClient:
    """Azure OpenAI client with rate limiting and error handling"""
    
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        api_version: str = "2024-02-15-preview",
        deployment_name: str = "gpt-4o",
        max_requests_per_minute: int = 60,
        max_tokens_per_minute: int = 150000,
        max_retries: int = 3
    ):
        self.client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute
        )
        self.max_retries = max_retries
        self.token_counter = TokenCounter(model="gpt-4o")
        
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 16000,
        temperature: float = 0.1,
        response_format: Optional[Dict] = None
    ) -> str:
        """Make chat completion request with rate limiting and retries"""
        
        # Estimate tokens
        prompt_text = "\n".join([m["content"] for m in messages])
        estimated_tokens = self.token_counter.count(prompt_text) + max_tokens
        
        # Acquire rate limit permission
        await self.rate_limiter.acquire(estimated_tokens)
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.deployment_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                if response_format:
                    kwargs["response_format"] = response_format
                
                response = await self.client.chat.completions.create(**kwargs)
                
                content = response.choices[0].message.content
                
                # Log token usage
                usage = response.usage
                logger.info(
                    f"API call successful. Tokens: prompt={usage.prompt_tokens}, "
                    f"completion={usage.completion_tokens}, total={usage.total_tokens}"
                )
                
                return content
                
            except RateLimitError as e:
                wait_time = 2 ** attempt
                logger.warning(f"Rate limit hit. Waiting {wait_time}s... (attempt {attempt + 1})")
                await asyncio.sleep(wait_time)
                
            except APIError as e:
                logger.error(f"API error: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
        
        raise Exception("Max retries exceeded")


class PageIndexTree:
    """
    Production PageIndex Tree Generator
    """
    
    def __init__(
        self,
        pdf_path: str,
        output_dir: str = "./pageindex_output",
        azure_endpoint: str = "",
        azure_api_key: str = "",
        deployment_name: str = "gpt-4o",
        max_pages_per_chunk: int = 10,
        max_tokens_per_chunk: int = 100000,  # Leave room for prompt overhead
        enable_disk_cache: bool = True,
        enable_summaries: bool = True,
        max_concurrent_requests: int = 5
    ):
        """Initialize PageIndex Tree Generator"""
        
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.pdf_extractor = PDFLayoutExtractor(pdf_path)
        self.llm_client = AzureOpenAIClient(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            deployment_name=deployment_name
        )
        self.token_counter = TokenCounter()
        
        # Configuration
        self.max_pages_per_chunk = max_pages_per_chunk
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.enable_disk_cache = enable_disk_cache
        self.enable_summaries = enable_summaries
        self.max_concurrent_requests = max_concurrent_requests
        
        # State
        self.root_nodes: List[DocumentNode] = []
        self.node_map: Dict[str, DocumentNode] = {}
        self.node_counter = 0
        self.checkpoint_file = self.output_dir / "checkpoint.pkl"
        
        # Concurrency control
        self.semaphore = Semaphore(max_concurrent_requests)
        
    def _generate_node_id(self) -> str:
        """Generate sequential node ID"""
        node_id = f"{self.node_counter:04d}"
        self.node_counter += 1
        return node_id
    
    def _save_checkpoint(self):
        """Save processing checkpoint to disk"""
        if not self.enable_disk_cache:
            return
        
        checkpoint = {
            "root_nodes": self.root_nodes,
            "node_map": self.node_map,
            "node_counter": self.node_counter,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Checkpoint saved: {len(self.node_map)} nodes")
    
    def _load_checkpoint(self) -> bool:
        """Load checkpoint if exists"""
        if not self.enable_disk_cache or not self.checkpoint_file.exists():
            return False
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.root_nodes = checkpoint["root_nodes"]
            self.node_map = checkpoint["node_map"]
            self.node_counter = checkpoint["node_counter"]
            
            logger.info(f"Checkpoint loaded: {len(self.node_map)} nodes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    async def _extract_structure_from_chunk(
        self,
        page_texts: List[str],
        start_page: int,
        end_page: int,
        previous_context: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Extract hierarchical structure from page chunk using LLM"""
        
        # Extract layout-based titles
        titles = self.pdf_extractor.extract_titles_from_range(start_page, end_page)
        
        # Build context
        title_context = "\n".join([
            f"Page {bbox.page_num}: '{text}' (Level {level}, font size ~{bbox.height:.1f})"
            for text, bbox, level in titles[:20]  # Limit to top 20 titles
        ])
        
        # Build page content
        page_content = ""
        for i, text in enumerate(page_texts):
            page_num = start_page + i
            page_content += f"\n\n=== PAGE {page_num} ===\n{text[:5000]}"  # Limit per-page text
        
        # Count tokens
        prompt_tokens = self.token_counter.count(page_content + title_context)
        logger.info(f"Chunk {start_page}-{end_page}: {prompt_tokens} prompt tokens")
        
        # Truncate if needed
        if prompt_tokens > self.max_tokens_per_chunk:
            page_content = self.token_counter.truncate(page_content, self.max_tokens_per_chunk - 2000)
            logger.warning(f"Truncated content to fit token limit")
        
        # Build prompt
        system_prompt = """You are an expert document structure analyzer. Extract the hierarchical structure of the document.

Output ONLY valid JSON in this exact format:
{
  "sections": [
    {
      "title": "Chapter 1: Introduction",
      "start_page": 1,
      "end_page": 5,
      "level": 0,
      "summary": "Brief 1-2 sentence summary"
    }
  ]
}

Rules:
- Use titles detected from layout (provided below)
- Level 0 = chapters, Level 1 = sections, Level 2 = subsections, etc.
- Be precise with page ranges
- Include ALL hierarchy levels
- Keep summaries under 150 words"""

        user_prompt = f"""Document: Pages {start_page}-{end_page}

Detected Titles (from layout analysis):
{title_context}

Page Content:
{page_content}

Extract the hierarchical structure as JSON:"""

        if previous_context:
            user_prompt = f"""Previous Structure (for context):
{json.dumps(previous_context[-3:], indent=2)}

{user_prompt}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call LLM
        async with self.semaphore:
            response = await self.llm_client.chat_completion(
                messages=messages,
                max_tokens=16000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
        
        # Parse response
        try:
            data = json.loads(response)
            sections = data.get("sections", [])
            logger.info(f"Extracted {len(sections)} sections from pages {start_page}-{end_page}")
            return sections
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response: {response[:500]}")
            return []
    
    async def _generate_summary(
        self,
        title: str,
        content: str,
        parent_summary: str = ""
    ) -> str:
        """Generate summary for a node"""
        
        if not self.enable_summaries:
            return ""
        
        # Truncate content
        content = self.token_counter.truncate(content, 10000)
        
        prompt = f"""Summarize the following document section in 2-3 sentences.

Title: {title}"""
        
        if parent_summary:
            prompt += f"\nParent Section: {parent_summary}"
        
        prompt += f"\n\nContent:\n{content}\n\nSummary:"
        
        messages = [
            {"role": "system", "content": "You are a technical document summarizer. Provide concise, accurate summaries."},
            {"role": "user", "content": prompt}
        ]
        
        async with self.semaphore:
            response = await self.llm_client.chat_completion(
                messages=messages,
                max_tokens=200,
                temperature=0.1
            )
        
        return response.strip()
    
    def _build_nodes_from_structure(
        self,
        structure: List[Dict],
        parent_node: Optional[DocumentNode] = None,
        parent_summary: str = ""
    ) -> List[DocumentNode]:
        """Build DocumentNode tree from LLM structure"""
        
        nodes = []
        
        for section in structure:
            # Extract page content
            start_page = section.get("start_page", 1)
            end_page = section.get("end_page", start_page)
            
            page_texts = []
            for p in range(start_page - 1, end_page):  # 0-indexed
                if p < self.pdf_extractor.total_pages:
                    page_texts.append(self.pdf_extractor.extract_page_text(p))
            
            text = "\n\n".join(page_texts)
            token_count = self.token_counter.count(text)
            
            # Create node
            node = DocumentNode(
                title=section.get("title", "Untitled"),
                node_id=self._generate_node_id(),
                start_page=start_page,
                end_page=end_page,
                text=text,
                summary=section.get("summary", ""),
                level=section.get("level", 0),
                parent_id=parent_node.node_id if parent_node else None,
                token_count=token_count,
                metadata={
                    "extracted_at": datetime.now().isoformat()
                }
            )
            
            # Store node
            self.node_map[node.node_id] = node
            nodes.append(node)
            
            # Process children recursively
            if "subsections" in section:
                node.children = self._build_nodes_from_structure(
                    section["subsections"],
                    parent_node=node,
                    parent_summary=node.summary
                )
            
            # Save intermediate result to disk
            if self.enable_disk_cache:
                node_file = self.output_dir / f"node_{node.node_id}.json"
                with open(node_file, 'w') as f:
                    json.dump(node.to_dict(), f, indent=2)
        
        return nodes
    
    async def generate_tree(self) -> Dict[str, Any]:
        """
        Main method: Generate PageIndex tree from PDF
        """
        
        logger.info(f"Starting PageIndex generation for: {self.pdf_path}")
        logger.info(f"Total pages: {self.pdf_extractor.total_pages}")
        
        # Check for existing checkpoint
        if self._load_checkpoint():
            logger.info("Resuming from checkpoint")
            return self.to_dict()
        
        # Split into chunks
        chunks = []
        for i in range(0, self.pdf_extractor.total_pages, self.max_pages_per_chunk):
            start = i
            end = min(i + self.max_pages_per_chunk - 1, self.pdf_extractor.total_pages - 1)
            
            # Extract texts
            page_texts = []
            for p in range(start, end + 1):
                page_texts.append(self.pdf_extractor.extract_page_text(p))
            
            chunks.append((page_texts, start + 1, end + 1))  # 1-indexed pages
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Process chunks iteratively
        all_sections = []
        previous_context = None
        
        for idx, (page_texts, start_page, end_page) in enumerate(chunks):
            logger.info(f"Processing chunk {idx + 1}/{len(chunks)}: pages {start_page}-{end_page}")
            
            try:
                sections = await self._extract_structure_from_chunk(
                    page_texts,
                    start_page,
                    end_page,
                    previous_context
                )
                
                if sections:
                    all_sections.extend(sections)
                    previous_context = sections
                
                # Periodic checkpoint
                if (idx + 1) % 5 == 0:
                    self._save_checkpoint()
                    
            except Exception as e:
                logger.error(f"Error processing chunk {idx + 1}: {e}")
                continue
        
        # Build tree
        logger.info("Building final tree structure...")
        self.root_nodes = self._build_nodes_from_structure(all_sections)
        
        # Generate summaries if needed
        if self.enable_summaries:
            logger.info("Generating node summaries...")
            await self._generate_all_summaries()
        
        # Final checkpoint
        self._save_checkpoint()
        
        logger.info(f"Tree generation complete. Total nodes: {len(self.node_map)}")
        
        return self.to_dict()
    
    async def _generate_all_summaries(self):
        """Generate summaries for all nodes without summaries"""
        
        tasks = []
        
        def collect_nodes(nodes):
            for node in nodes:
                if not node.summary and node.text:
                    tasks.append((node, self._generate_summary(
                        node.title,
                        node.text,
                        ""
                    )))
                if node.children:
                    collect_nodes(node.children)
        
        collect_nodes(self.root_nodes)
        
        logger.info(f"Generating {len(tasks)} summaries...")
        
        # Process in batches
        batch_size = self.max_concurrent_requests
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            summaries = await asyncio.gather(*[task for _, task in batch])
            
            for (node, _), summary in zip(batch, summaries):
                node.summary = summary
            
            logger.info(f"Generated summaries {i + 1}-{min(i + batch_size, len(tasks))}/{len(tasks)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to PageIndex format"""
        
        return {
            "success": True,
            "doc_name": Path(self.pdf_path).name,
            "total_pages": self.pdf_extractor.total_pages,
            "total_nodes": len(self.node_map),
            "structure": [node.to_dict() for node in self.root_nodes],
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model": self.llm_client.deployment_name,
                "max_pages_per_chunk": self.max_pages_per_chunk,
                "summaries_enabled": self.enable_summaries
            }
        }
    
    def save_json(self, filepath: Optional[str] = None):
        """Save tree to JSON file"""
        
        if filepath is None:
            filepath = self.output_dir / "pageindex_tree.json"
        else:
            filepath = Path(filepath)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Tree saved to: {filepath}")
    
    def print_tree(self, nodes: Optional[List[DocumentNode]] = None, indent: int = 0):
        """Print tree structure"""
        
        if nodes is None:
            nodes = self.root_nodes
        
        for node in nodes:
            prefix = "  " * indent
            print(f"{prefix}[{node.node_id}] {node.title}")
            print(f"{prefix}  └─ Pages: {node.start_page}-{node.end_page} | Tokens: {node.token_count}")
            
            if node.summary:
                summary = node.summary[:100] + "..." if len(node.summary) > 100 else node.summary
                print(f"{prefix}     {summary}")
            
            if node.children:
                self.print_tree(node.children, indent + 1)
    
    def cleanup(self):
        """Cleanup resources"""
        self.pdf_extractor.close()


# Main execution
async def main():
    """Example usage"""
    
    # Configuration
    PDF_PATH = "your_document.pdf"
    OUTPUT_DIR = "./pageindex_output"
    
    # Azure OpenAI credentials (set these from environment or config)
    AZURE_ENDPOINT = "https://your-resource.openai.azure.com/"
    AZURE_API_KEY = "your-api-key"
    DEPLOYMENT_NAME = "gpt-4o"
    
    # Initialize generator
    generator = PageIndexTree(
        pdf_path=PDF_PATH,
        output_dir=OUTPUT_DIR,
        azure_endpoint=AZURE_ENDPOINT,
        azure_api_key=AZURE_API_KEY,
        deployment_name=DEPLOYMENT_NAME,
        max_pages_per_chunk=10,
        max_tokens_per_chunk=100000,
        enable_disk_cache=True,
        enable_summaries=True,
        max_concurrent_requests=5
    )
    
    try:
        # Generate tree
        result = await generator.generate_tree()
        
        # Print structure
        print("\n" + "=" * 80)
        print("PAGEINDEX TREE STRUCTURE")
        print("=" * 80)
        generator.print_tree()
        
        # Save to JSON
        generator.save_json()
        
        print("\n" + "=" * 80)
        print(f"SUCCESS: Generated {result['total_nodes']} nodes for {result['total_pages']} pages")
        print(f"Output saved to: {OUTPUT_DIR}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Failed to generate tree: {e}", exc_info=True)
        raise
    
    finally:
        generator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())