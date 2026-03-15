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
  - **Note:** Kuzu may be swapped out for a different graphdb 
- **Multi-Agent Support**: Isolated graphs per agent prevent false confidence from shared associations
- **Configurable Parameters**: Tune exploration rate, decay rate, and strength thresholds for different use cases  

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

### How Associations Guide Agent Behavior

Resonance provides agents with learned context about what concepts connect in the user's mind. This enables more intelligent, personalized interactions.

#### Without Memory (Cold Start Every Time)
```
User: "Help me with Python"
Agent: "Sure! What would you like to do with Python?"
User: "Data analysis"
Agent: "What kind of data?"
User: "The usual pandas stuff"
Agent: "Which pandas functions?"
```
Every interaction starts from zero context.

#### With Resonance Memory
```
User: "Help me with Python"

Associations retrieved:
  python → data_science [0.92]
  python → pandas [0.85]
  python → jupyter [0.73]

Agent: "I can help with your Python data science work. 
       Are you working in pandas again, or trying something new?"
```

The agent:
- **Makes informed assumptions** (probably data science, not web development)
- **Asks better questions** (specific to pandas, not generic)  
- **Skips redundant clarification** (knows the user's typical context)
- **Retrieves smarter** (if using web_search, searches "python pandas" not just "python")

#### Confidence Scores Drive Behavior

**Strong associations (>0.7)**: Safe to assume
```python
# python → pandas [0.92]
# Agent can confidently suggest pandas-specific solutions
```

**Medium associations (0.3-0.7)**: Worth mentioning
```python
# python → plotly [0.45]  
# Agent might ask: "Need visualization with plotly?"
```

**Weak associations (<0.3)**: Exploratory only
```python
# python → django [0.15]
# Agent ignores unless user brings it up
```

#### Multi-Tool Integration

Associations improve other tool usage:

**Web Search:**
```python
# User asks about "authentication"
# Associations: authentication → oauth [0.88], authentication → jwt [0.76]
# Agent searches: "oauth jwt authentication" (not generic "authentication")
```

**Code Generation:**
```python
# User asks to "add error handling"  
# Associations: error_handling → logging [0.91], error_handling → try_except [0.87]
# Agent includes both patterns automatically
```

**Document Retrieval:**
```python
# User mentions "the project"
# Associations: project → customer_dashboard [0.95]
# Agent retrieves customer_dashboard docs, not all projects
```

#### The Result

Agents that feel like they **know you** instead of treating every interaction as the first time you've met.

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

## Configuration Guidance

### Multiple Agents

**Different agents should use different graph paths:**

Each agent type should maintain its own separate graph to avoid confusion from shared associations.
```python
# Chatbot agent
chatbot = ResonanceMemory(
    graph_path='./chatbot_graph'
)

# Code review agent
reviewer = ResonanceMemory(
    graph_path='./reviewer_graph'
)

# Research assistant
researcher = ResonanceMemory(
    graph_path='./researcher_graph'
)
```

**Why separate graphs?**
- Prevents Agent B from inheriting strong associations from Agent A's experience
- Avoids false confidence in relationships one agent never learned
- Each agent builds its own understanding through its own interactions

### Exploration Rate Tuning

The `exploration_rate` parameter (default: 0.2) controls exploitation vs exploration:

**Low Exploration (0.05 - 0.1)**: Reliable, consistent
- Customer support agents
- Task-focused assistants
- Production systems where reliability matters

**Medium Exploration (0.15 - 0.25)**: Balanced (default)
- General purpose assistants
- Personal productivity agents
- Most use cases

**High Exploration (0.3 - 0.5)**: Creative, experimental
- Research assistants
- Creative writing tools
- Discovery-focused applications

### Common Configurations

**Personal Assistant:**
```python
memory = ResonanceMemory(
    graph_path='./personal_assistant',
    exploration_rate=0.2,
    decay_rate=0.005,  # Slower decay, longer memory
    min_strength=0.15   # Higher threshold, only strong associations
)
```

**Customer Support:**
```python
memory = ResonanceMemory(
    graph_path='./customer_support',
    exploration_rate=0.1,  # More consistent
    decay_rate=0.02,       # Faster decay, recent issues more relevant
    min_strength=0.1
)
```

**Research/Discovery:**
```python
memory = ResonanceMemory(
    graph_path='./research_agent',
    exploration_rate=0.35,  # More exploration
    decay_rate=0.01,
    min_strength=0.05       # Lower threshold, surface weak connections
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

## Integration Example

Resonance is designed to integrate seamlessly into any agent loop. Here's how to add persistent memory to your agent:
```python
from resonance import ResonanceMemory

# Initialize memory for your agent
memory = ResonanceMemory(
    graph_path='./my_agent_memory',
    exploration_rate=0.2
)

# Your agent loop
while True:
    # Get user input
    user_input = input("User: ")
    
    # 1. RECALL - Get relevant context from memory
    context = memory.recall(user_input)
    
    # 2. GENERATE - Use associations to enrich your response
    # Pass context['associations'] to your agent/LLM
    response = your_agent.generate(
        prompt=user_input,
        associations=context['associations'],
        semantic_context=context['semantic_context']
    )
    
    print(f"Agent: {response}")
    
    # 3. REMEMBER - Store this interaction
    memory.remember_interaction(user_input, response)
    memory.increment_generation()
```

### What Gets Passed to Your Agent

The `recall()` method returns:
```python
{
    'concepts': ['elephant', 'justice'],  # Extracted from query
    'associations': {
        'elephant': [
            ('shollublip', 0.95),  # concept, confidence
            ('large', 0.82)
        ]
    },
    'semantic_context': [...],  # Related past conversations
    'generation': 1205  # Current cycle count
}
```

Use these associations to:
- Inform your prompt construction
- Provide context about user preferences
- Surface related topics from past conversations
- Build continuity across sessions

### Interpreting Confidence Scores

Associations come with confidence scores that indicate strength:
- **High (>0.7)**: Strong established pattern - the user frequently connects these concepts
- **Medium (0.3-0.7)**: Moderate connection - mentioned together sometimes  
- **Low (<0.3)**: Weak or exploratory association - rarely connected

When building your system prompt, explain to your agent that higher scores represent stronger learned patterns. The agent should use strong associations to inform its responses naturally, without explicitly saying "I remember that..." unless contextually appropriate.


### LLM Integration Example
```python
# Example with Anthropic API
import anthropic

client = anthropic.Anthropic()
memory = ResonanceMemory(graph_path='./assistant')

def chat(user_input):
    # Get memory context
    context = memory.recall(user_input)
    
    # Build enriched prompt
    associations_str = "\n".join([
        f"- {concept}: {', '.join([f'{assoc[0]} ({assoc[1]:.2f})' for assoc in assocs])}"
        for concept, assocs in context['associations'].items()
    ])
    
    system_prompt = f"""You are a helpful assistant with memory of past conversations.

Relevant associations from memory:
{associations_str}

These associations show concepts this user frequently connects together. The confidence scores indicate strength:
- Scores above 0.7 are strong, established patterns
- Scores 0.3-0.7 are moderate connections
- Scores below 0.3 are weak, exploratory links

Use these to provide contextual, personalized responses that reflect the user's patterns and preferences. Reference these connections naturally without explicitly mentioning "my memory" unless directly relevant."""
    
    # Generate
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": user_input}],
        system=system_prompt
    )
    
    agent_response = response.content[0].text
    
    # Remember
    memory.remember_interaction(user_input, agent_response)
    memory.increment_generation()
    
    return agent_response
```

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

No - but you're welcome to rip it off and improve it 

## Acknowledgments

Built with:
- [Kuzu](https://kuzudb.com/) - Embedded graph database
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [spaCy](https://spacy.io/) - NLP and concept extraction

## Further Reading

For those interested in the academic foundations of associative memory for LLM agents:

**LLM Associative Memory Agents (LAMA)** - Inoshita (2026)
- Paper: [arXiv:2601.12771](https://arxiv.org/abs/2601.12771)
- Explores nationality prediction using recall-based reasoning with dual-agent architecture
- Demonstrates that LLMs are more reliable at recalling concrete examples than abstract reasoning
- Published January 2026

Resonance extends similar associative memory concepts to general-purpose agent memory with temporal decay mechanics inspired by cognitive science.
