#!/usr/bin/env python3
"""
Obsidian Vault → Resonance Memory Ingestion

Converts your Obsidian notes into a Resonance memory graph.
Preserves your wikilinks as explicit associations and extracts
concepts from all your note content.

Usage:
    python ingest_obsidian.py /path/to/obsidian/vault ./resonance_obsidian
"""

import os
import sys
import re
from pathlib import Path
from resonance import ResonanceMemory


def parse_markdown_file(filepath):
    """
    Extract content and wikilinks from a markdown file.
    
    Returns:
        (content, wikilinks, title)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract title (filename without extension)
    title = Path(filepath).stem
    
    # Find all [[wikilinks]] and [[wikilinks|aliases]]
    wikilinks = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)
    
    # Clean up content - remove excessive whitespace but keep structure
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content, wikilinks, title


def ingest_obsidian_vault(vault_path, memory_path='./resonance_obsidian', debug=False):
    """
    Ingest an entire Obsidian vault into Resonance memory.
    
    Args:
        vault_path: Path to your Obsidian vault directory
        memory_path: Where to store the Resonance graph
        debug: Print debug info during ingestion
    """
    vault_path = Path(vault_path)
    
    if not vault_path.exists():
        print(f"Error: Vault path '{vault_path}' does not exist")
        return
    
    # Initialize memory
    print(f"Initializing Resonance memory at: {memory_path}")
    memory = ResonanceMemory(
        graph_path=memory_path,
        debug=debug
    )
    
    # Find all markdown files
    md_files = list(vault_path.rglob('*.md'))
    print(f"\nFound {len(md_files)} markdown files")
    
    if len(md_files) == 0:
        print("No markdown files found. Check your vault path.")
        return
    
    # Process each file
    notes_processed = 0
    associations_created = 0
    
    for md_file in md_files:
        try:
            content, wikilinks, title = parse_markdown_file(md_file)
            
            # Skip empty files
            if not content.strip():
                continue
            
            if debug:
                print(f"\n📄 Processing: {title}")
                print(f"   Links: {wikilinks}")
            
            # Store the note content
            # Use title as "user" and content as "agent" so concepts get extracted from both
            result = memory.remember_interaction(
                user_input=f"Note titled: {title}",
                agent_response=content
            )
            
            associations_created += result['associations_created']
            
            # Create explicit associations for wikilinks
            # Each wikilink creates a strong connection
            for link in wikilinks:
                # Force association between note title and linked concept
                link_result = memory.remember_interaction(
                    user_input=title,
                    agent_response=f"{link} is related"
                )
                associations_created += link_result['associations_created']
            
            memory.increment_generation()
            notes_processed += 1
            
            if notes_processed % 10 == 0:
                print(f"  Processed {notes_processed}/{len(md_files)} notes...")
                
        except Exception as e:
            print(f"⚠️  Error processing {md_file}: {e}")
            continue
    
    print(f"\n✅ Ingestion Complete!")
    print(f"   Notes processed: {notes_processed}")
    print(f"   Associations created: {associations_created}")
    print(f"   Final generation: {memory.get_generation()}")
    print(f"\nMemory saved to: {memory_path}")
    print(f"\nYou can now use this with:")
    print(f"   memory = ResonanceMemory(graph_path='{memory_path}')")


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest_obsidian.py <vault_path> [memory_path] [--debug]")
        print("\nExample:")
        print("  python ingest_obsidian.py ~/Documents/ObsidianVault ./my_memory")
        sys.exit(1)
    
    vault_path = sys.argv[1]
    memory_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else './resonance_obsidian'
    debug = '--debug' in sys.argv
    
    print("🧠 Obsidian → Resonance Ingestion")
    print("=" * 50)
    
    ingest_obsidian_vault(vault_path, memory_path, debug)


if __name__ == '__main__':
    main()