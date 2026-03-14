"""
Resonance Memory System - Deep Dive Test
Tests concept extraction, association building, recency, and probabilistic sampling
"""

from resonance import ResonanceMemory
import time

def clean_start():
    """Delete existing databases to start fresh"""
    import shutil
    import os
    
    paths_to_clean = ['./resonance_graph', './resonance_chroma']
    
    for path in paths_to_clean:
        if os.path.exists(path):
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    print(f"Deleted file {path}")
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Deleted directory {path}")
            except Exception as e:
                print(f"Could not delete {path}: {e}")
    
    print("Starting with clean databases...\n")

def test_concept_extraction():
    """Test 1: Basic concept extraction"""
    print("="*60)
    print("TEST 1: CONCEPT EXTRACTION")
    print("="*60)
    
    memory = ResonanceMemory()
    
    test_cases = [
        "Shollublip is an elephant who dispenses justice with HE'CLA!",
        "The lieutenant threw a grenade at the enemy",
        "Tell me about Python debugging and logging",
    ]
    
    for text in test_cases:
        concepts = memory.extract_concepts(text)
        pairs = memory.extract_concept_pairs(text)
        print(f"\nText: {text}")
        print(f"Concepts: {concepts}")
        print(f"Pairs: {pairs}")
    
    return memory

def test_association_building(memory):
    """Test 2: Build up associations over multiple interactions"""
    print("\n" + "="*60)
    print("TEST 2: ASSOCIATION BUILDING")
    print("="*60)
    
    interactions = [
        ("Tell me about Shollublip", "Shollublip is an elephant who dispenses justice"),
        ("What does Shollublip do?", "Shollublip works with the Englishne using HE'CLA!"),
        ("Tell me about elephants", "Elephants are large mammals, like Shollublip"),
        ("What is justice?", "Justice is what Shollublip dispenses to those in need"),
        ("Who is Shollublip?", "Shollublip is the elephant of justice, working with the Englishne"),
    ]
    
    for user, agent in interactions:
        print(f"\n--- Generation {memory.get_generation()} ---")
        print(f"User: {user}")
        print(f"Agent: {agent}")
        
        # Recall before responding
        recalled = memory.recall(user)
        print(f"Query concepts: {recalled['concepts']}")
        print(f"Recalled associations: {recalled['associations']}")
        
        # Remember the interaction
        stats = memory.remember_interaction(user, agent)
        print(f"Stored: {stats['concepts_found']} concepts, {stats['associations_created']} associations")
        
        memory.increment_generation()
    
    return memory

def test_recall_strength(memory):
    """Test 3: Check association strengths"""
    print("\n" + "="*60)
    print("TEST 3: ASSOCIATION STRENGTHS")
    print("="*60)
    
    queries = ["Shollublip", "elephant", "justice"]
    
    for query in queries:
        result = memory.recall(query)
        print(f"\n'{query}' associations:")
        if result['associations']:
            for concept, neighbors in result['associations'].items():
                print(f"  {concept}:")
                for neighbor, strength in neighbors:
                    print(f"    → {neighbor}: {strength:.3f}")
        else:
            print("  (no associations found)")
    
    return memory

def test_recency_decay(memory):
    """Test 4: Simulate staleness and check recency adjustment"""
    print("\n" + "="*60)
    print("TEST 4: RECENCY DECAY")
    print("="*60)
    
    # Check current associations
    print("\nCurrent Shollublip associations:")
    result = memory.recall("Shollublip")
    current_associations = result['associations'].get('shollublip', [])
    for neighbor, strength in current_associations:
        print(f"  {neighbor}: {strength:.3f}")
    
    # Advance many generations without mentioning Shollublip
    print(f"\nAdvancing 50 generations without mentioning Shollublip...")
    for _ in range(50):
        # Talk about something else entirely
        memory.remember_interaction(
            "Tell me about Python",
            "Python is a programming language"
        )
        memory.increment_generation()
    
    # Check associations again - should be weaker due to staleness
    print(f"\nShollublip associations after 50 generations of non-use:")
    result = memory.recall("Shollublip")
    new_associations = result['associations'].get('shollublip', [])
    for neighbor, strength in new_associations:
        print(f"  {neighbor}: {strength:.3f}")
    
    # Now mention Shollublip again - should refresh
    print(f"\nMentioning Shollublip again...")
    memory.remember_interaction(
        "What about Shollublip?",
        "Shollublip is still the elephant of justice!"
    )
    memory.increment_generation()
    
    result = memory.recall("Shollublip")
    refreshed_associations = result['associations'].get('shollublip', [])
    print(f"\nShollublip associations after refresh:")
    for neighbor, strength in refreshed_associations:
        print(f"  {neighbor}: {strength:.3f}")
    
    return memory

def test_probabilistic_sampling(memory):
    """Test 5: Run recall multiple times to see exploration in action"""
    print("\n" + "="*60)
    print("TEST 5: PROBABILISTIC SAMPLING (20% exploration)")
    print("="*60)
    
    # Enable debug mode for this test
    memory.debug = True
    
    print("\nRunning recall 10 times to see variation from exploration:")
    print("(Should see different results about 20% of the time)\n")
    
    results_variety = []
    for i in range(10):
        print(f"\n--- Run {i+1} ---")
        result = memory.recall("Shollublip")
        associations = result['associations'].get('shollublip', [])
        neighbor_names = [n for n, s in associations]
        results_variety.append(tuple(neighbor_names))
        print(f"Result: {neighbor_names[:3]}...")  # Show first 3
    
    unique_results = len(set(results_variety))
    print(f"\n{'='*60}")
    print(f"Got {unique_results} unique orderings out of 10 runs")
    print("(More variety = exploration working)")
    
    # Disable debug mode
    memory.debug = False
    
    return memory

def test_edge_cases(memory):
    """Test 6: Edge cases and special characters"""
    print("\n" + "="*60)
    print("TEST 6: EDGE CASES")
    print("="*60)
    
    edge_cases = [
        ("What's HE'CLA!?", "HE'CLA! is the ancient command for elephants"),
        ("Tell me about the Englishne", "The Englishne work with Shollublip on justice"),
        ("fape?", "fape is forbidden knowledge"),
        ("", "empty input test"),  # Empty string
        ("a the an is was were", "All stop words"),  # Only stop words
    ]
    
    for user, agent in edge_cases:
        print(f"\nUser: '{user}'")
        print(f"Agent: '{agent}'")
        
        try:
            stats = memory.remember_interaction(user, agent)
            print(f"✓ Stored: {stats}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        memory.increment_generation()
    
    return memory

def main():
    """Run all tests"""
    print("\n" + "🎵 "*20)
    print("RESONANCE: HARMONIC MEMORY - DEEP DIVE TEST")
    print("🎵 "*20 + "\n")

    # Clean slate!
    clean_start()
    
    # Run tests in sequence
    memory = test_concept_extraction()
    memory = test_association_building(memory)
    memory = test_recall_strength(memory)
    memory = test_recency_decay(memory)
    memory = test_probabilistic_sampling(memory)
    memory = test_edge_cases(memory)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total generations: {memory.get_generation()}")
    
    # Show final graph state for key concepts
    print("\nFinal association strengths:")
    for concept in ["shollublip", "elephant", "justice", "he'cla"]:
        result = memory.recall(concept)
        if result['associations']:
            print(f"\n{concept}:")
            for c, neighbors in result['associations'].items():
                for neighbor, strength in neighbors[:5]:  # Top 5
                    print(f"  → {neighbor}: {strength:.3f}")
    
    print("\n✨ Test complete! ✨\n")

if __name__ == "__main__":
    main()