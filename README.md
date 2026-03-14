# Resonance: Harmonic Memory for LLM Agents

A persistent associative memory system that enables LLM agents to learn and remember relationships across conversations. Resonance uses graph-based storage with harmonic decay mechanics to build a memory that strengthens with use and naturally fades when unused.

## Overview

Most LLM agents either have no memory between sessions or rely on simple retrieval (RAG). Resonance provides something different: **associative memory** that learns what concepts relate to each other based on how often they appear together, with confidence that grows logarithmically over time.

### Key Features

- **Logarithmic Growth**: Associations strengthen with repeated mentions, with diminishing returns over time
- **Recency Adjustment**: Memories fade based on how many generation cycles have passed since last use
- **Probabilistic Sampling**: 20% exploration rate prevents echo chambers by occasionally surfacing weaker associations
- **Lemmatization**: Automatically handles plurals and verb forms ("elephants" → "elephant")
- **Persistent Storage**: All data saved locally using Kuzu (graph) and ChromaDB (semantic search)

## Installation
```bash
# Clone or download the repository
cd resonance

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

## Quick Start
```python
from resonance import ResonanceMemory

# Initialize memory system
memory = ResonanceMemory()

# In your agent loop:
user_input = "Tell me about elephants"

# 1. Recall relevant memories before responding
context = memory.recall(user_input)
print(f"Associations: {context['associations']}")

# 2. Generate your response (using your LLM)
agent_response = "Elephants are large mammals..."

# 3. Store the interaction
memory.remember_interaction(user_input, agent_response)

# 4. Increment generation counter
memory.increment_generation()
```

## How It Works

### Memory Mechanics

**Association Strength:**
- New associations start weak (~0.1)
- Strengthen logarithmically: `new_strength = base + log(1 + activations) * increment`
- Cap at maximum strength (1.0)

**Recency Adjustment:**
When retrieving memories, strength is adjusted based on staleness:
```
adjusted_strength = base_strength × e^(-decay_rate × cycles_elapsed / base_strength)
```

Strong memories resist decay better than weak ones.

**Exploration vs Exploitation:**
- 80% of the time: Returns strongest associations (exploitation)
- 20% of the time: Samples probabilistically (exploration)

This prevents the system from getting locked into rigid patterns while still being reliable.

### Example: Building Associations
```python
memory = ResonanceMemory()

# First mention
memory.remember_interaction(
    "Tell me about Shollublip",
    "Shollublip is an elephant"
)
memory.increment_generation()
# Creates: shollublip ↔ elephant [strength: 0.1]

# Second mention
memory.remember_interaction(
    "What does Sholluplip do?",
    "Shollublip dispenses justice"
)
memory.increment_generation()
# Strengthens: shollublip ↔ elephant [strength: 0.17]
# Creates: shollublip ↔ justice [strength: 0.1]

# After many mentions
# shollublip ↔ elephant [strength: 0.95] - very confident
# shollublip ↔ justice [strength: 0.87] - confident
```

## Configuration

All parameters are configurable:
```python
memory = ResonanceMemory(
    graph_path='./my_graph',              # Where to store graph database
    chroma_path='./my_chroma',            # Where to store vector database
    chroma_collection_name='my_memory',   # Collection name in Chroma
    exploration_rate=0.2,                 # 20% exploration
    min_strength=0.1,                     # Minimum association threshold
    max_strength=1.0,                     # Maximum association cap
    decay_rate=0.01,                      # Rate of recency decay
    spacy_model='en_core_web_sm',        # spaCy model for concept extraction
    debug=False                           # Enable debug output
)
```

## API Reference

### ResonanceMemory

#### `__init__(**config)`
Initialize the memory system with optional configuration.

#### `recall(query: str, max_associations: int = 20) -> Dict`
Retrieve relevant memories for a query.

**Returns:**
```python
{
    'concepts': ['elephant', 'justice'],
    'associations': {
        'elephant': [
            ('shollublip', 0.95),
            ('mammal', 0.82)
        ]
    },
    'semantic_context': [...],  # Related past interactions
    'generation': 42
}
```

#### `remember_interaction(user_input: str, agent_response: str) -> Dict`
Store associations from a conversation turn.

**Returns:**
```python
{
    'concepts_found': 5,
    'associations_created': 10
}
```

#### `increment_generation() -> int`
Advance the generation counter. Call after each interaction.

#### `get_generation() -> int`
Get current generation count.

#### `extract_concepts(text: str) -> List[str]`
Extract concepts from text (useful for debugging).

## Use Cases

### Personal Assistant
Build a map of user's projects, interests, and relationships. Strong edges form for frequent collaborators, weak edges for one-off interactions.

### Customer Support
Remember customer preferences and past issues. Associations between problems and solutions strengthen when they work repeatedly.

### Code Assistant
Learn user's coding patterns and architecture preferences. "When working on auth, also consider logging and error handling."

### Research Assistant
Build citation networks and concept clusters. Relationships strengthen as papers are discussed together.

## Why Not Just RAG?

RAG retrieves similar documents. Resonance learns **relationships**:

| Feature | RAG | Resonance |
|---------|-----|-----------|
| Query | "Find documents about elephants" | "Elephant strongly associates with Sholluplip (0.95) and justice (0.87)" |
| Learning | Static similarity | Learned confidence from usage |
| Memory | No relationship memory | Remembers what connects to what |
| Variety | Same results every time | Explores new connections 20% of the time |

## Testing

Run the comprehensive test suite:
```bash
python test_resonance.py
```

Tests cover:
- Concept extraction (including lemmatization)
- Association building with logarithmic growth
- Recency decay over time
- Probabilistic sampling (exploration vs exploitation)
- Edge cases (special characters, empty strings, etc.)

## Performance

- **Graph queries**: < 10ms for typical association lookups
- **Concept extraction**: ~50-100ms per interaction (spaCy processing)
- **Storage**: Minimal (thousands of concepts ≈ 1MB)

## Technical Details

### Storage

**Kuzu Graph Database:**
- Stores concepts as nodes
- Stores associations as edges with:
  - `base_strength`: Core association strength
  - `last_accessed_generation`: When last used
  - `total_activations`: How many times reinforced

**ChromaDB Vector Database:**
- Stores full interaction text for semantic search
- Enables finding conceptually similar past conversations

### Concept Extraction

Uses spaCy with:
- Named entity recognition
- Noun chunk extraction
- Lemmatization (handles plurals, verb forms)
- Case normalization

All concepts stored as lowercase lemmas for consistent matching.

## License

MIT

## Contributing

Issues and pull requests welcome!

## Acknowledgments

Built with:
- [Kuzu](https://kuzudb.com/) - Embedded graph database
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [spaCy](https://spacy.io/) - NLP and concept extraction