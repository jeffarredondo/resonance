"""
Resonance: Harmonic Memory for LLM Agents

A persistent associative memory system that learns from interactions.
Combines semantic search (Chroma) with graph-based associations (Kuzu)
to provide context-aware memory recall with exploration.
"""

import kuzu
import chromadb
import spacy
import pandas as pd
import random
import math
from typing import List, Dict, Tuple, Optional
from itertools import combinations


class ResonanceMemory:
    """
    Harmonic memory system for LLM agents.
    
    Provides two main operations:
    1. recall() - Retrieve relevant memories and associations
    2. remember_interaction() - Store new associations from agent interactions
    
    Memory Mechanics:
    - New associations start weak (~0.1)
    - Strengthen logarithmically with each activation
    - Recency-adjusted on retrieval (old unused memories weaken)
    - Probabilistic sampling ensures exploration of weak associations
    
    Usage:
        memory = ResonanceMemory()
        
        # Before generating response
        context = memory.recall("user asked about elephants")
        
        # After generating response
        memory.remember_interaction(
            user_input="Tell me about elephants",
            agent_response="Elephants are large mammals..."
        )
        memory.increment_generation()
    """
    
    def __init__(
        self, 
        graph_path: str = './resonance_graph',
        chroma_path: str = './resonance_chroma',
        chroma_collection_name: str = 'resonance_memory',
        exploration_rate: float = 0.2,
        min_strength: float = 0.1,
        max_strength: float = 1.0,
        decay_rate: float = 0.01,
        spacy_model: str = "en_core_web_sm",
        debug: bool = False
    ):
        """
        Initialize the memory system.
        
        Args:
            graph_path: Path to Kuzu graph database
            chroma_path: Path to Chroma vector database
            chroma_collection_name: Name for the Chroma collection
            exploration_rate: Probability of exploring weak associations (default 0.2)
            min_strength: Minimum association strength threshold
            max_strength: Maximum association strength cap
            decay_rate: Rate at which memories decay based on staleness
            spacy_model: spaCy model to use for concept extraction
            debug: Enable debug output
        """
        self.exploration_rate = exploration_rate
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.decay_rate = decay_rate
        self.debug = debug
        
        # Initialize spaCy for concept extraction
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading spaCy model: {spacy_model}")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)
        
        # Initialize Kuzu graph
        self.db = kuzu.Database(graph_path)
        self.conn = kuzu.Connection(self.db)
        self._initialize_schema()
        
        # Initialize Chroma
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=chroma_collection_name
        )
    
    def _initialize_schema(self):
        """Create database schema if it doesn't exist."""
        try:
            # Try to query SystemState - if it fails, schema doesn't exist
            self.conn.execute("MATCH (s:SystemState) RETURN s LIMIT 1")
        except:
            # Schema doesn't exist, create it
            print("Initializing Resonance schema...")
            
            self.conn.execute("""
                CREATE NODE TABLE Concept(
                    name STRING,
                    first_seen_generation INT64,
                    PRIMARY KEY(name)
                )
            """)
            
            self.conn.execute("""
                CREATE REL TABLE ASSOCIATES_WITH(
                    FROM Concept TO Concept,
                    base_strength DOUBLE,
                    last_accessed_generation INT64,
                    total_activations INT64
                )
            """)
            
            self.conn.execute("""
                CREATE NODE TABLE SystemState(
                    key STRING,
                    value INT64,
                    PRIMARY KEY(key)
                )
            """)
            
            # Initialize generation counter
            self.conn.execute("""
                CREATE (s:SystemState {key: 'generation_count', value: 0})
            """)
            
            print("Schema initialized successfully!")
    
    def extract_concepts(self, text: str) -> List[str]:
        """
        Extract meaningful concepts from text using spaCy with lemmatization.
        
        Extracts:
        - Named entities (PERSON, ORG, GPE, etc.)
        - Noun chunks (noun phrases)
        
        Filters out:
        - Pronouns, determiners, stop words
        - Leading articles (the, a, an)
        
        All concepts are lemmatized (base form) and normalized to lowercase.
        
        Args:
            text: Input text to extract concepts from
            
        Returns:
            List of concept strings (lowercase, lemmatized)
            
        Example:
            >>> extract_concepts("Elephants are large mammals")
            ['elephant', 'large', 'mammal']  # Note: 'elephants' → 'elephant'
        """
        doc = self.nlp(text)
        concepts = set()
        
        # Get named entities - use lemma for base form
        for ent in doc.ents:
            # For multi-word entities, lemmatize each token
            lemmatized = ' '.join([token.lemma_.lower() for token in ent])
            concepts.add(lemmatized)
        
        # Get noun chunks (noun phrases)
        for chunk in doc.noun_chunks:
            # Skip if it's just a pronoun or determiner
            if chunk.root.pos_ not in ['PRON', 'DET']:
                # Use the lemma of the root word
                lemma = chunk.root.lemma_.lower()
                
                # Clean up: remove if it's an article
                if lemma not in ['the', 'a', 'an'] and lemma.strip():
                    concepts.add(lemma)
        
        return list(concepts)
    
    def extract_concept_pairs(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract all pairs of concepts that co-occur in text.
        
        Args:
            text: Input text
            
        Returns:
            List of (concept_a, concept_b) tuples
            
        Example:
            >>> extract_concept_pairs("Shollublip is an elephant")
            [('shollublip', 'elephant')]
        """
        concepts = self.extract_concepts(text)
        
        # Create all pairs (combinations, not permutations)
        pairs = list(combinations(concepts, 2))
        
        return pairs
    
    def recall(
        self, 
        query: str, 
        max_associations: int = 20,
        semantic_results: int = 5
    ) -> Dict:
        """
        Retrieve relevant memories for a given query.
        
        Combines semantic search (finds conceptually similar content) with
        graph traversal (finds associated concepts). Uses probabilistic sampling
        to balance exploitation (strong memories) vs exploration (weak memories).
        
        Args:
            query: Natural language query to search for
            max_associations: Maximum number of associations to return per concept
            semantic_results: Number of semantic search results to retrieve
            
        Returns:
            {
                'semantic_context': List of relevant text snippets from past interactions,
                'associations': Dict mapping concepts to their associated concepts with strengths,
                'concepts': List of concepts extracted from query,
                'generation': Current generation count
            }
            
        Example:
            >>> memory.recall("what do elephants do?")
            {
                'semantic_context': [...],
                'concepts': ['elephants'],
                'associations': {
                    'elephant': [
                        ('sholluplip', 0.95),
                        ('large', 0.82),
                    ]
                },
                'generation': 1205
            }
        """
        current_gen = self._get_generation()
        
        # 1. Extract concepts from query
        query_concepts = self.extract_concepts(query)
        
        if self.debug:
            print(f"[DEBUG] Extracted concepts from query: {query_concepts}")
        
        # 2. Semantic search in Chroma
        semantic_matches = self._semantic_search(query, semantic_results)
        
        # 3. Graph traversal to find associations
        associations = {}
        for concept in query_concepts:
            neighbors = self._get_neighbors_with_sampling(
                concept, 
                current_gen, 
                max_associations
            )
            if neighbors:
                associations[concept] = neighbors
        
        return {
            'semantic_context': semantic_matches,
            'concepts': query_concepts,
            'associations': associations,
            'generation': current_gen
        }
    
    def remember_interaction(
        self,
        user_input: str,
        agent_response: str,
        strength_increment: float = 0.1
    ) -> Dict[str, int]:
        """
        Store associations from a user-agent interaction.
        
        Extracts concepts from both user input and agent response,
        then creates/strengthens associations between co-occurring concepts.
        
        Args:
            user_input: What the user said
            agent_response: What the agent responded
            strength_increment: Amount to increase strength per activation
            
        Returns:
            Statistics about what was stored:
            {
                'concepts_found': number of unique concepts,
                'associations_created': number of new/strengthened associations
            }
        """
        current_gen = self._get_generation()
        
        # Combine both texts for concept extraction
        combined_text = f"{user_input} {agent_response}"
        
        # Extract concept pairs
        pairs = self.extract_concept_pairs(combined_text)
        
        if self.debug:
            print(f"[DEBUG] Extracted {len(pairs)} concept pairs from interaction")
        
        # Store each pair
        for concept_a, concept_b in pairs:
            self._ensure_concept_exists(concept_a, current_gen)
            self._ensure_concept_exists(concept_b, current_gen)
            self._add_or_strengthen_edge(
                concept_a, 
                concept_b, 
                current_gen, 
                strength_increment
            )
        
        # Store full interaction in Chroma for semantic search
        self._store_context(combined_text, current_gen)
        
        # Get unique concepts
        all_concepts = self.extract_concepts(combined_text)
        
        return {
            'concepts_found': len(all_concepts),
            'associations_created': len(pairs)
        }
    
    def increment_generation(self) -> int:
        """
        Advance the generation counter.
        
        Should be called after each agent interaction/generation cycle.
        Used for recency calculations.
        
        Returns:
            New generation count
        """
        current = self._get_generation()
        new_gen = current + 1
        
        self.conn.execute(
            """
            MATCH (s:SystemState {key: 'generation_count'})
            SET s.value = $new_gen
            """,
            parameters={'new_gen': new_gen}
        )
        
        return new_gen
    
    def get_generation(self) -> int:
        """Get current generation count."""
        return self._get_generation()
    
    # ========== Internal Methods ==========
    
    def _get_generation(self) -> int:
        """Retrieve current generation count from graph."""
        result = self.conn.execute(
            "MATCH (s:SystemState {key: 'generation_count'}) RETURN s.value"
        )
        rows = list(result)
        return rows[0][0] if rows else 0
    
    def _semantic_search(self, query: str, n_results: int) -> List[str]:
        """Query Chroma for semantically similar content."""
        try:
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results['documents'][0] if results['documents'] else []
        except:
            return []
    
    def _get_neighbors_with_sampling(
        self, 
        concept: str, 
        current_gen: int, 
        max_results: int
    ) -> List[Tuple[str, float]]:
        """
        Get associated concepts with probabilistic sampling.
        
        80% of time: Return top-K strongest (exploitation)
        20% of time: Sample probabilistically by strength (exploration)
        
        All strengths are recency-adjusted before selection.
        """
        neighbors = self._get_neighbors_with_recency(concept, current_gen)
        
        if not neighbors:
            return []
        
        # Decide: exploit or explore?
        explore = random.random() < self.exploration_rate
        
        if self.debug:
            mode = "EXPLORING" if explore else "EXPLOITING"
            print(f"  [DEBUG] {mode} for '{concept}' ({len(neighbors)} neighbors available)")
        
        if explore:
            # EXPLORE: Probabilistic sampling
            names = [n['name'] for n in neighbors]
            weights = [n['adjusted_strength'] for n in neighbors]
            
            k = min(max_results, len(neighbors))
            sampled_indices = random.choices(range(len(neighbors)), weights=weights, k=k)
            sampled = [neighbors[i] for i in set(sampled_indices)]
            
            if self.debug:
                print(f"  [DEBUG] Sampled {len(sampled)} neighbors probabilistically")
        else:
            # EXPLOIT: Top-K strongest
            sampled = neighbors[:max_results]
            
            if self.debug:
                print(f"  [DEBUG] Took top {len(sampled)} strongest neighbors")
        
        return [(n['name'], n['adjusted_strength']) for n in sampled]
    
    def _get_neighbors_with_recency(
        self, 
        concept: str, 
        current_gen: int
    ) -> List[Dict]:
        """
        Get neighbors with recency-adjusted strengths.
        
        Applies exponential decay based on staleness:
        adjusted_strength = base_strength * exp(-decay_rate * staleness / base_strength)
        
        Stronger memories resist decay better than weak ones.
        """
        result = self.conn.execute(
            """
            MATCH (c:Concept {name: $name})-[r:ASSOCIATES_WITH]-(neighbor)
            RETURN neighbor.name as name,
                   r.base_strength as base_strength,
                   r.last_accessed_generation as last_gen,
                   r.total_activations as activations
            """,
            parameters={'name': concept}
        )
        
        neighbors = []
        
        for row in result:
            staleness = current_gen - row[2]
            base_strength = row[1]
            
            # Recency adjustment: stronger memories decay slower
            adjusted_strength = base_strength * math.exp(
                -self.decay_rate * staleness / base_strength
            )
            
            if adjusted_strength >= self.min_strength:
                neighbors.append({
                    'name': row[0],
                    'base_strength': base_strength,
                    'adjusted_strength': adjusted_strength,
                    'activations': row[3]
                })
        
        neighbors.sort(key=lambda x: x['adjusted_strength'], reverse=True)
        return neighbors
    
    def _ensure_concept_exists(self, name: str, generation: int) -> None:
        """Create concept node if it doesn't exist."""
        self.conn.execute(
            """
            MERGE (c:Concept {name: $name})
            ON CREATE SET c.first_seen_generation = $gen
            """,
            parameters={'name': name, 'gen': generation}
        )
    
    def _add_or_strengthen_edge(
        self, 
        concept_a: str, 
        concept_b: str, 
        generation: int,
        increment: float
    ) -> None:
        """
        Add new association or strengthen existing one.
        
        New: Creates edge with initial strength
        Existing: Applies logarithmic growth
            new_strength = min(base + log(1 + activations) * increment, max_strength)
        """
        # Check if edge exists
        try:
            result = self.conn.execute(
                """
                MATCH (a:Concept {name: $name_a})-[r:ASSOCIATES_WITH]-(b:Concept {name: $name_b})
                RETURN r.base_strength as strength, r.total_activations as activations
                """,
                parameters={'name_a': concept_a, 'name_b': concept_b}
            )
            
            existing = list(result)
        except:
            existing = []
        
        if existing:
            # Strengthen existing edge with logarithmic growth
            current_strength = existing[0][0]
            current_activations = existing[0][1]
            
            # Logarithmic growth: diminishing returns over time
            new_strength = min(
                current_strength + math.log(1 + current_activations) * increment,
                self.max_strength
            )
            new_activations = current_activations + 1
            
            if self.debug:
                print(f"  [DEBUG] Strengthening '{concept_a}' ↔ '{concept_b}': {current_strength:.3f} → {new_strength:.3f}")
            
            try:
                self.conn.execute(
                    """
                    MATCH (a:Concept {name: $name_a})-[r:ASSOCIATES_WITH]-(b:Concept {name: $name_b})
                    SET r.base_strength = $strength,
                        r.last_accessed_generation = $gen,
                        r.total_activations = $activations
                    """,
                    parameters={
                        'name_a': concept_a,
                        'name_b': concept_b,
                        'strength': new_strength,
                        'gen': generation,
                        'activations': new_activations
                    }
                )
            except Exception as e:
                if self.debug:
                    print(f"  [DEBUG] Error strengthening edge: {e}")
        else:
            # Create new edge - but first verify both nodes exist
            try:
                verify = self.conn.execute(
                    """
                    MATCH (a:Concept {name: $name_a})
                    MATCH (b:Concept {name: $name_b})
                    RETURN count(*) as cnt
                    """,
                    parameters={'name_a': concept_a, 'name_b': concept_b}
                )
                count = list(verify)[0][0]
                
                if count == 0:
                    if self.debug:
                        print(f"  [DEBUG] Nodes don't exist for '{concept_a}' ↔ '{concept_b}', skipping")
                    return
            except:
                if self.debug:
                    print(f"  [DEBUG] Could not verify nodes exist, skipping edge creation")
                return
            
            # Now safe to create edge
            if self.debug:
                print(f"  [DEBUG] Creating new edge '{concept_a}' ↔ '{concept_b}': {increment:.3f}")
            
            try:
                self.conn.execute(
                    """
                    MATCH (a:Concept {name: $name_a}), (b:Concept {name: $name_b})
                    CREATE (a)-[r:ASSOCIATES_WITH {
                        base_strength: $strength,
                        last_accessed_generation: $gen,
                        total_activations: 1
                    }]->(b)
                    """,
                    parameters={
                        'name_a': concept_a,
                        'name_b': concept_b,
                        'strength': increment,
                        'gen': generation
                    }
                )
            except Exception as e:
                if self.debug:
                    print(f"  [DEBUG] Error creating edge: {e}")
    
    def _store_context(self, context: str, generation: int) -> None:
        """Store interaction context in Chroma for semantic search."""
        try:
            self.chroma_collection.add(
                documents=[context],
                metadatas=[{'generation': generation}],
                ids=[f"gen_{generation}"]
            )
        except Exception:
            pass