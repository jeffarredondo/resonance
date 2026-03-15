# A complete Obsidian ingestion script

To use it:

```bash
# Basic usage
python ingest_obsidian.py /path/to/your/ObsidianVault

# Specify where to save the memory
python ingest_obsidian.py /path/to/your/ObsidianVault ./my_obsidian_memory

# Debug mode to see what's happening
python ingest_obsidian.py /path/to/your/ObsidianVault --debug
```

What it does:
- Finds all .md files in your vault
- Extracts concepts from note content
- Preserves [[wikilinks]] as explicit associations
- Handles [[link|alias]] syntax
- Skips empty files
- Shows progress every 10 files

Then you can use it:

```python
from resonance import ResonanceMemory

# Load your Obsidian-powered memory
memory = ResonanceMemory(graph_path='./resonance_obsidian')

# Now your agent has access to ALL your notes!
context = memory.recall("What did I write about machine learning?")
```
