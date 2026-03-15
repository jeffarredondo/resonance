# Memento vs RAG vs Resonance: Understanding Memory Architectures for LLMs

## TL;DR

| Approach | What It Does | Best For | Limitation |
|----------|-------------|----------|------------|
| **Memento** | Fine-tunes model to use in-context examples better | Improving coherence in small models | Persists in weights but no decay; expensive to update |
| **RAG** | Retrieves similar documents to add to context | Finding relevant information from knowledge base | No relationship learning, static |
| **Resonance** | Builds association graph that strengthens with use | Long-term relationship memory across sessions | Requires interaction to build up |

## Memento: Teaching Models to Use Context

### What It Is
A fine-tuning approach that trains models to better utilize in-context examples. Based on the research paper showing that models can learn to reference and build on previous conversation turns more effectively.

### How It Works
```
Training Data:
- Conversation turn 1
- Conversation turn 2  
- Conversation turn 3 (references 1 and 2)

Model learns: "When I see patterns like turn 1 and 2, I should connect them in turn 3"
```

### Strengths
- Improves coherence within a single conversation
- Helps small models compete with larger ones on conversational tasks
- No external infrastructure needed once trained
- Works entirely within model weights

### Limitations
- **No persistence**: Forgets everything when conversation ends
- **Requires fine-tuning**: Need training data and compute
- **Context window limited**: Can only reference what fits in context
- **No cross-session learning**: Each conversation starts fresh

### Use Case
Making a small local model (like smaLLM) maintain coherent conversations without needing massive parameter counts.

---

## RAG: Retrieval Augmented Generation

### What It Is
Semantic search over a document/chunk database. Find relevant text snippets and inject them into the prompt.

### How It Works
```
User: "What do elephants eat?"

1. Embed query → [0.23, 0.45, 0.12, ...]
2. Search vector DB for similar embeddings
3. Retrieve top-K chunks:
   - "Elephants are herbivores..."
   - "Their diet consists of grass, leaves..."
4. Add to prompt:
   Context: [retrieved chunks]
   User question: What do elephants eat?
   
Model generates answer using retrieved context
```

### Strengths
- **Access to external knowledge**: Not limited to training data
- **Easy to update**: Add new documents without retraining
- **Semantic matching**: Finds conceptually similar content
- **Works with any model**: No fine-tuning needed

### Limitations
- **No relationship learning**: Doesn't learn what connects to what
- **Static similarity**: Same query always returns same results
- **No confidence signals**: Can't tell if "elephants → herbivore" is strongly established or weakly mentioned
- **No temporal awareness**: Doesn't know if information is stale
- **Chunk-based**: Loses context that spans multiple chunks

### Use Case
Adding a knowledge base to an agent (documentation, company policies, research papers).

---

## Resonance: Harmonic Associative Memory

### What It Is
A graph-based memory system that learns associations between concepts through repeated exposure, with strength that grows logarithmically and decays with disuse.

### How It Works
```
User: "Tell me about elephants"

1. Extract concepts: ['elephant']
2. Query graph for associations:
   elephant --0.95--> shollublip (strong, frequently mentioned together)
   elephant --0.72--> herbivore (moderate, mentioned together sometimes)
   elephant --0.23--> africa (weak, rarely mentioned together)
3. Probabilistically sample:
   - 80%: Return top strongest (exploitation)
   - 20%: Sample by weight (exploration)
4. Return: [('sholluplip', 0.95), ('herbivore', 0.72)]

Agent uses these associations to inform response

After response:
5. Extract concepts from both user and agent messages
6. Create/strengthen edges between co-occurring concepts
7. Increment generation counter (for recency tracking)
```

### Strengths
- **Learns relationships**: Builds up "X relates to Y" with confidence scores
- **Temporal awareness**: Recent associations stronger, old ones decay
- **Exploration**: Occasionally tries weaker associations (prevents echo chambers)
- **Persistent**: Survives across sessions, weeks, months
- **Confidence signals**: 0.95 means "very sure", 0.2 means "might be related"
- **Self-organizing**: No manual curation needed

### Limitations
- **Requires interactions to build**: Starts empty, needs usage to become useful
- **Not a knowledge base**: Stores relationships, not facts
- **Concept extraction quality**: Depends on NLP (spaCy) accuracy
- **No reasoning**: Just associations, doesn't validate truth

### Use Case
Long-term agent memory that learns user patterns, project relationships, and domain connections over time.

---

## Comparison Scenarios

### Scenario 1: User asks "What's Python good for?"

**Memento (fine-tuned model):**
- Uses in-context examples from current conversation
- Might reference "you mentioned data science earlier"
- Forgets after session ends

**RAG:**
- Searches knowledge base: "Python is a programming language..."
- Returns factual documentation
- Same answer every time

**Resonance:**
- Checks associations: python --0.87--> data_science, python --0.65--> scripting
- Knows "this user always discusses Python in data science context" (learned from history)
- Might explore weaker association (python --0.3--> game_dev) 20% of the time

---

### Scenario 2: 50 conversations later, user asks "What was that library I liked?"

**Memento:**
- Can't help - no memory of 50 conversations ago
- Would need entire conversation history in context (impossible)

**RAG:**
- Searches past conversations if stored
- Returns chunks mentioning libraries
- No sense of "which one you REALLY liked" vs mentioned once

**Resonance:**
- Shows: pandas --0.95--> user (mentioned constantly)
- Shows: numpy --0.78--> user (mentioned often)  
- Shows: scipy --0.12--> user (mentioned once)
- Confidence signals tell you: "Definitely pandas"

---

### Scenario 3: Teaching an agent your coding style

**Memento:**
- Within a conversation, maintains consistency
- Next session: Forgets your style preferences

**RAG:**
- Retrieves past code examples
- No learning of patterns ("user prefers X when doing Y")

**Resonance:**
- Learns: authentication --0.92--> logging (you always add logging to auth code)
- Learns: error_handling --0.88--> try_except_else (your preferred pattern)
- Learns: testing --0.76--> pytest (your framework choice)
- Agent: "You're working on auth. Should I include logging like usual?"

---

## Combining Them

The best systems use multiple approaches:

### Small Local Model (like smaLLM)
```
Base Model
  ↓
+ Memento fine-tuning (better in-context coherence)
  ↓
+ Resonance (long-term associative memory)
```

### API-Based Agent (Claude, GPT)
```
Base Model (already good at using context)
  ↓
+ RAG (knowledge base access)
  ↓
+ Resonance (relationship learning)
```

### Full Stack
```
Query arrives
  ↓
1. Resonance: "What concepts relate to this query?"
  ↓
2. RAG: "Retrieve documents about those concepts"
  ↓
3. Model (with Memento): Generate response using both
  ↓
4. Resonance: Store new associations from response
```

---

## When to Use What

### Use Memento When:
- You have a small model that struggles with coherence
- You can afford fine-tuning compute
- You want better in-conversation performance
- You don't need cross-session memory
- [link to paper](https://arxiv.org/abs/2508.16153)

### Use RAG When:
- You need to access a knowledge base
- Information changes frequently (docs, policies)
- You want semantic search over documents
- You need factual retrieval

### Use Resonance When:
- You want long-term relationship memory
- You're building personal assistants or agents
- You want the system to learn user patterns
- You need confidence signals on associations
- You want exploration of new connections

### Use All Three When:
- You're building a production agent that needs:
  - Coherent conversations (Memento)
  - Access to knowledge (RAG)
  - Long-term relationship learning (Resonance)

---

## The Lieutenant & Grenade Problem

A perfect example of why Resonance matters:

**User memory:** "I always thought 'lieutenant' and 'grenade' went together because I watched China Beach as a kid."

**RAG:** Would find documents mentioning both, but wouldn't know they're strongly associated *in your mind*

**Resonance:** 
```
lieutenant --0.85--> grenade (strong personal association)
```

When you mention "lieutenant" in any context, Resonance surfaces "grenade" because *for you*, they're connected. Even though it makes no logical sense, it's a true memory pattern.

**This is how human memory actually works.** Resonance captures that.

---

## Summary

- **Memento** = Better in-context learning (within a conversation)
- **RAG** = Access to external knowledge (documents, facts)
- **Resonance** = Long-term associative memory (relationships, patterns, confidence)

They solve different problems. The best systems use them together.