#!/usr/bin/env python3
"""
Example usage of PageIndex Tree Generator
Demonstrates complete workflow with proper error handling
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv
import logging

# Import the PageIndex tree generator
from pageindex_tree import PageIndexTree, logger


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)


def validate_environment():
    """Validate required environment variables"""
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT"
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   - {var}")
        print("\nPlease set them in .env file or environment")
        sys.exit(1)
    
    print("✓ Environment configuration validated")


async def process_document(
    pdf_path: str,
    output_dir: str = None,
    max_pages: int = None,
    enable_summaries: bool = True,
    resume: bool = True,
    verbose: bool = False
):
    """
    Process a PDF document and generate PageIndex tree
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory (default: ./pageindex_output)
        max_pages: Process only first N pages (None = all)
        enable_summaries: Generate summaries for nodes
        resume: Resume from checkpoint if exists
        verbose: Enable verbose logging
    """
    
    setup_logging(verbose)
    
    # Validate inputs
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if output_dir is None:
        output_dir = f"./pageindex_output_{pdf_path.stem}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("PAGEINDEX TREE GENERATION")
    print("=" * 80)
    print(f"PDF: {pdf_path}")
    print(f"Output: {output_dir}")
    print(f"Summaries: {'Enabled' if enable_summaries else 'Disabled'}")
    print(f"Resume: {'Enabled' if resume else 'Disabled'}")
    print("=" * 80 + "\n")
    
    # Initialize generator
    generator = PageIndexTree(
        pdf_path=str(pdf_path),
        output_dir=str(output_dir),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        max_pages_per_chunk=int(os.getenv("MAX_PAGES_PER_CHUNK", "10")),
        max_tokens_per_chunk=int(os.getenv("MAX_TOKENS_PER_CHUNK", "100000")),
        enable_disk_cache=resume,
        enable_summaries=enable_summaries,
        max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    )
    
    # Check if resuming
    checkpoint_exists = (output_dir / "checkpoint.pkl").exists()
    if checkpoint_exists and resume:
        print("📂 Found existing checkpoint - will resume processing\n")
    elif checkpoint_exists and not resume:
        print("⚠️  Checkpoint exists but --no-resume specified - starting fresh\n")
        (output_dir / "checkpoint.pkl").unlink()
    
    try:
        # Generate tree
        print("🚀 Starting tree generation...\n")
        start_time = asyncio.get_event_loop().time()
        
        result = await generator.generate_tree()
        
        elapsed_time = asyncio.get_event_loop().time() - start_time
        
        # Save results
        output_file = output_dir / "pageindex_tree.json"
        generator.save_json(str(output_file))
        
        # Print summary
        print("\n" + "=" * 80)
        print("GENERATION COMPLETE")
        print("=" * 80)
        print(f"✓ Total pages: {result['total_pages']}")
        print(f"✓ Total nodes: {result['total_nodes']}")
        print(f"✓ Processing time: {elapsed_time:.1f}s")
        print(f"✓ Output file: {output_file}")
        print("=" * 80 + "\n")
        
        # Print tree structure
        print("📊 DOCUMENT STRUCTURE:\n")
        generator.print_tree()
        
        # Print statistics
        print("\n" + "=" * 80)
        print("NODE STATISTICS")
        print("=" * 80)
        
        # Calculate statistics
        total_tokens = sum(node.token_count for node in generator.node_map.values())
        avg_tokens = total_tokens / len(generator.node_map) if generator.node_map else 0
        
        level_counts = {}
        for node in generator.node_map.values():
            level_counts[node.level] = level_counts.get(node.level, 0) + 1
        
        print(f"Total tokens processed: {total_tokens:,}")
        print(f"Average tokens per node: {avg_tokens:.0f}")
        print("\nNodes per level:")
        for level in sorted(level_counts.keys()):
            print(f"  Level {level}: {level_counts[level]} nodes")
        
        print("=" * 80 + "\n")
        
        # Cost estimation (GPT-4o pricing as of 2024)
        input_cost = (total_tokens / 1_000_000) * 2.50
        output_cost = (result['total_nodes'] * 200 / 1_000_000) * 10.00  # Estimate 200 tokens per summary
        total_cost = input_cost + output_cost
        
        print("💰 ESTIMATED COST (GPT-4o):")
        print(f"  Input tokens: ~{total_tokens:,} → ${input_cost:.2f}")
        print(f"  Output tokens: ~{result['total_nodes'] * 200:,} → ${output_cost:.2f}")
        print(f"  Total: ~${total_cost:.2f}")
        print("  (Check pageindex.log for exact token usage)\n")
        
        return result
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print(f"💾 Checkpoint saved to: {output_dir / 'checkpoint.pkl'}")
        print("   Run again with same parameters to resume\n")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Failed to generate tree: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print(f"📋 Check logs: {output_dir.parent / 'pageindex.log'}\n")
        sys.exit(1)
        
    finally:
        generator.cleanup()


def analyze_existing_tree(json_path: str):
    """Analyze an existing PageIndex tree JSON"""
    
    import json
    
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"❌ JSON file not found: {json_path}")
        sys.exit(1)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\n" + "=" * 80)
    print("PAGEINDEX TREE ANALYSIS")
    print("=" * 80)
    print(f"Document: {data['doc_name']}")
    print(f"Total pages: {data['total_pages']}")
    print(f"Total nodes: {data['total_nodes']}")
    print(f"Generated: {data['metadata'].get('generated_at', 'Unknown')}")
    print("=" * 80 + "\n")
    
    # Analyze structure
    def analyze_nodes(nodes, level=0):
        for node in nodes:
            indent = "  " * level
            page_range = f"{node['start_index']}-{node['end_index']}"
            print(f"{indent}[{node['node_id']}] {node['title']} (pp. {page_range})")
            
            if 'summary' in node:
                summary = node['summary'][:80] + "..." if len(node['summary']) > 80 else node['summary']
                print(f"{indent}  └─ {summary}")
            
            if 'nodes' in node:
                analyze_nodes(node['nodes'], level + 1)
    
    analyze_nodes(data['structure'])
    
    print("\n" + "=" * 80)


def main():
    """Main CLI entry point"""
    
    parser = argparse.ArgumentParser(
        description="PageIndex Tree Generator - Convert PDFs to hierarchical document structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python example_usage.py document.pdf
  
  # Custom output directory
  python example_usage.py document.pdf -o ./my_output
  
  # Disable summaries for faster processing
  python example_usage.py document.pdf --no-summaries
  
  # Process first 50 pages only
  python example_usage.py document.pdf --max-pages 50
  
  # Analyze existing tree
  python example_usage.py --analyze output/pageindex_tree.json
  
  # Verbose logging
  python example_usage.py document.pdf -v
"""
    )
    
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Path to PDF file"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: ./pageindex_output_<filename>)"
    )
    
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Process only first N pages"
    )
    
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="Disable summary generation (faster)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--analyze",
        metavar="JSON_PATH",
        help="Analyze existing PageIndex tree JSON"
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Handle analyze mode
    if args.analyze:
        analyze_existing_tree(args.analyze)
        return
    
    # Validate PDF path
    if not args.pdf_path:
        parser.print_help()
        print("\n❌ Error: PDF path required (or use --analyze)")
        sys.exit(1)
    
    # Validate environment
    validate_environment()
    
    # Process document
    asyncio.run(process_document(
        pdf_path=args.pdf_path,
        output_dir=args.output,
        max_pages=args.max_pages,
        enable_summaries=not args.no_summaries,
        resume=not args.no_resume,
        verbose=args.verbose
    ))


if __name__ == "__main__":
    main()