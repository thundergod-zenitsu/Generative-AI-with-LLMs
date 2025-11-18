# PageIndex Tree Generator

Production-grade implementation of PageIndex document structure extraction with Azure OpenAI. Converts PDF documents into hierarchical trees with layout-aware title detection, summaries, and bounding box metadata.

## 🚀 Features

- ✅ **Azure OpenAI Integration** - Fully tested with GPT-4o (128K context, 16K output)
- ✅ **Layout Detection** - Extracts titles from actual PDF layout (font sizes, bounding boxes)
- ✅ **Token-Based Chunking** - Uses tiktoken for precise token counting
- ✅ **Rate Limiting** - Sophisticated rate limiter with request & token tracking
- ✅ **Resumable Processing** - Disk-based checkpoints for large documents
- ✅ **Memory Efficient** - Writes intermediate nodes to disk
- ✅ **Parallel Processing** - Concurrent API calls with configurable limits
- ✅ **Comprehensive Logging** - Track token usage and costs
- ✅ **Error Handling** - Automatic retries, graceful degradation

## 📋 Requirements

- Python 3.8+
- Azure OpenAI account with GPT-4o deployment
- PDF documents (text-based, not scanned images)

## 🔧 Installation

### 1. Clone/Download the files

```bash
# Download all files:
# - pageindex_tree.py (main implementation)
# - example_usage.py (CLI tool)
# - requirements.txt
# - .env.example
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Azure OpenAI

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` with your Azure credentials:

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

**To get your credentials:**
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Azure OpenAI resource
3. Click "Keys and Endpoint" in the sidebar
4. Copy the endpoint URL and one of the keys
5. Note your deployment name from "Deployments" section

## 🎯 Quick Start

### Basic Usage

```bash
python example_usage.py document.pdf
```

This will:
- Process the entire PDF
- Generate hierarchical structure
- Create summaries for each section
- Save results to `./pageindex_output_document/`

### Custom Output Directory

```bash
python example_usage.py document.pdf -o ./my_output
```

### Fast Mode (No Summaries)

```bash
python example_usage.py document.pdf --no-summaries
```

### Process First 50 Pages

```bash
python example_usage.py document.pdf --max-pages 50
```

### Analyze Existing Tree

```bash
python example_usage.py --analyze output/pageindex_tree.json
```

### Resume Interrupted Processing

The generator automatically saves checkpoints. If interrupted, just run the same command again:

```bash
python example_usage.py document.pdf -o ./output
# ... interrupted ...
python example_usage.py document.pdf -o ./output  # Resumes automatically
```

## 📊 Output Format

### JSON Structure (PageIndex-compatible)

```json
{
  "success": true,
  "doc_name": "document.pdf",
  "total_pages": 150,
  "total_nodes": 45,
  "structure": [
    {
      "title": "Chapter 1: Introduction",
      "node_id": "0000",
      "start_index": 1,
      "end_index": 15,
      "summary": "This chapter introduces fundamental concepts...",
      "token_count": 12450,
      "bbox": {
        "x0": 72.0,
        "y0": 100.5,
        "x1": 540.0,
        "y1": 120.8,
        "page": 1
      },
      "nodes": [
        {
          "title": "1.1 Background",
          "node_id": "0001",
          "start_index": 2,
          "end_index": 5,
          "summary": "Background section covers...",
          "bbox": {...}
        }
      ]
    }
  ],
  "metadata": {
    "generated_at": "2025-11-18T10:30:00",
    "model": "gpt-4o",
    "max_pages_per_chunk": 10
  }
}
```

### Output Files

```
pageindex_output/
├── checkpoint.pkl           # Resume checkpoint
├── node_0000.json          # Individual node files
├── node_0001.json
├── pageindex_tree.json     # Complete tree
└── pageindex.log           # Detailed logs
```

## 🔌 Programmatic Usage

```python
import asyncio
import os
from dotenv import load_dotenv
from pageindex_tree import PageIndexTree

load_dotenv()

async def main():
    generator = PageIndexTree(
        pdf_path="document.pdf",
        output_dir="./output",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name="gpt-4o",
        max_pages_per_chunk=10,
        enable_disk_cache=True,
        enable_summaries=True,
        max_concurrent_requests=5
    )
    
    # Generate tree
    result = await generator.generate_tree()
    
    # Save to JSON
    generator.save_json("output/tree.json")
    
    # Access nodes
    for node in generator.root_nodes:
        print(f"{node.title}: {node.summary}")
    
    # Cleanup
    generator.cleanup()

asyncio.run(main())
```

## ⚙️ Configuration

### Rate Limiting

Adjust based on your Azure quota (check in Azure Portal):

```bash
# .env file
MAX_REQUESTS_PER_MINUTE=60      # API calls per minute
MAX_TOKENS_PER_MINUTE=150000    # Tokens per minute
MAX_CONCURRENT_REQUESTS=5       # Parallel requests
```

The system automatically throttles to stay within limits.

### Chunking Strategy

```bash
MAX_PAGES_PER_CHUNK=10          # Pages per API call
MAX_TOKENS_PER_CHUNK=100000     # Max tokens per chunk (留space for prompt)
```

**Trade-offs:**
- **Smaller chunks** (5 pages): Better structure detection, more API calls
- **Larger chunks** (20 pages): Fewer API calls, may miss details

### Memory Efficiency

For very large documents (1000+ pages):

```python
generator = PageIndexTree(
    pdf_path="large_doc.pdf",
    enable_disk_cache=True,        # Write nodes to disk
    max_pages_per_chunk=5,         # Smaller chunks
    max_tokens_per_chunk=50000     # Less memory
)
```

## 💰 Cost Estimation

### GPT-4o Pricing (November 2024)
- Input: $2.50 / 1M tokens
- Output: $10.00 / 1M tokens

### Example Costs

| Document Size | Est. Input Tokens | Est. Output Tokens | Est. Cost |
|---------------|-------------------|--------------------|-----------:|
| 50 pages      | ~50K              | ~5K                | ~$0.18     |
| 200 pages     | ~200K             | ~20K               | ~$0.70     |
| 500 pages     | ~500K             | ~50K               | ~$1.75     |
| 1000 pages    | ~1M               | ~100K              | ~$3.50     |

**Check exact usage:**
- Monitor `pageindex.log` for precise token counts
- Each API call logs: `prompt_tokens`, `completion_tokens`, `total_tokens`

## 📈 Performance

### Processing Speed

| Document Size | Time (5 workers) | Time (10 workers) |
|---------------|------------------|-------------------|
| 50 pages      | ~2 minutes       | ~1.5 minutes      |
| 200 pages     | ~8 minutes       | ~5 minutes        |
| 500 pages     | ~20 minutes      | ~12 minutes       |
| 1000 pages    | ~40 minutes      | ~25 minutes       |

*Times vary based on network latency and document complexity*

### Optimization Tips

1. **Increase concurrent requests** (if quota allows):
   ```python
   max_concurrent_requests=10
   ```

2. **Disable summaries** for faster processing:
   ```python
   enable_summaries=False
   ```

3. **Adjust chunk size** based on document type:
   - Dense academic papers: `max_pages_per_chunk=5`
   - Reports/documentation: `max_pages_per_chunk=15`

## 🐛 Troubleshooting

### Rate Limit Errors

```
WARNING - Rate limit hit. Waiting 30.5s...
```

**Solution:** Reduce concurrency or increase quota
```bash
MAX_CONCURRENT_REQUESTS=3
```

### Token Limit Exceeded

```
WARNING - Truncated content to fit token limit
```

**Solution:** Reduce chunk size
```bash
MAX_PAGES_PER_CHUNK=5
MAX_TOKENS_PER_CHUNK=50000
```

### JSON Parsing Errors

```
ERROR - Failed to parse LLM response
```

**Solutions:**
1. Ensure model supports JSON mode (GPT-4o does)
2. Check API version in `.env`
3. The system automatically retries

### PDF Extraction Issues

**Problem:** Blank or garbled text

**Solutions:**
1. Ensure PDF is text-based (not scanned)
2. For scanned PDFs, use OCR preprocessing
3. Check PDF isn't encrypted/protected

### Out of Memory

For very large documents:

```python
# Use smaller chunks and enable disk cache
generator = PageIndexTree(
    pdf_path="huge_doc.pdf",
    enable_disk_cache=True,
    max_pages_per_chunk=3,
    max_tokens_per_chunk=30000
)
```

## 📝 Logging

The system logs everything to `pageindex.log`:

```bash
# Watch in real-time
tail -f pageindex.log

# Search for errors
grep ERROR pageindex.log

# Find token usage
grep "Tokens:" pageindex.log
```

## 🔒 Security Notes

- Store `.env` file securely (never commit to Git)
- Add `.env` to `.gitignore`
- Rotate API keys periodically
- Use Azure RBAC for production deployments

## 🧪 Testing

Test with a small document first:

```bash
# Process only first 10 pages
python example_usage.py large_doc.pdf --max-pages 10 -v
```

Review output quality before processing entire document.

## 🤝 Common Use Cases

### Academic Papers
```bash
python example_usage.py paper.pdf -o ./paper_analysis
```

### Technical Documentation
```bash
python example_usage.py manual.pdf --no-summaries -o ./manual_structure
```

### Legal Documents
```bash
python example_usage.py contract.pdf -o ./contract_analysis
```

### Books/Textbooks
```bash
# Process in batches for very long books
python example_usage.py textbook.pdf --max-pages 200 -o ./textbook_part1
```

## 📚 Advanced Topics

### Custom Prompts

Modify prompts in `pageindex_tree.py`:

```python
def _build_structure_prompt(self, ...):
    system_prompt = """Your custom instructions here..."""
    # ...
```

### Layout Detection Tuning

Adjust title detection heuristics:

```python
@property
def is_likely_title(self) -> bool:
    return (
        self.font_size > 14 and  # Adjust threshold
        len(self.text) < 300 and  # Longer titles
        (self.is_bold or self.text.isupper())
    )
```

### Hierarchical Level Detection

Customize level assignment:

```python
@property
def hierarchy_level(self) -> int:
    if self.font_size >= 24:    # Main chapters
        return 0
    elif self.font_size >= 18:  # Sections
        return 1
    # ...
```

## 📖 API Reference

See [Configuration & Usage Guide](config_guide.md) for complete API documentation.

## 🙏 Acknowledgments

Inspired by [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) - the original reasoning-based document indexing system.

## 📄 License

This implementation is provided as-is for educational and research purposes.

---

**Questions or Issues?** Check the logs first, then review the troubleshooting guide above.