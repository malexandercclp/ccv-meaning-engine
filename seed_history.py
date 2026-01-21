"""
CCV Complex Life History Seed

Creates a realistic, interconnected meaning structure representing
a 35-year-old person's lived experience for testing the CCV system.

The seeded person:
- Grew up with supportive mother, emotionally distant father
- Parents divorced when they were 12
- Found solace in books and music as a child
- Struggled with anxiety in adolescence
- Went to college, studied psychology
- Had a significant relationship that ended painfully
- Now works as a therapist
- Has a dog named Luna
- Practices meditation
- Has complicated feelings about success and self-worth

Run: python -m ccv_v1.seed_history
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from core import (
    CCVProcessor, RecurrenceBuffer,
    VALENCE_POSITIVE, VALENCE_NEGATIVE, VALENCE_NEUTRAL,
    get_valence_tagged_axis, setup_logging
)


def seed_complex_history():
    """Seed a complex life history into the buffer."""
    
    # Setup logging
    logger = setup_logging("INFO")
    
    processor = CCVProcessor()
    buffer = processor.buffer
    
    # Clear existing data
    buffer.clear(create_backup=True)
    
    print("=" * 60)
    print("SEEDING COMPLEX LIFE HISTORY")
    print("=" * 60)
    
    # Define experiences with structure:
    # (object_name, interrogatives_dict, crystallizations_dict, sentiment)
    experiences = [
        # ===== EARLY CHILDHOOD (Foundation) =====
        
        # Mother - warm, supportive
        ("mother", {
            "what+": "parent / caregiver",
            "why+": "loved me / protected me / read to me / made me feel safe",
            "when+": "childhood / always there",
            "where+": "home"
        }, {"what+": 0.85, "why+": 0.9}, 0.7),
        
        # Father - emotionally distant
        ("father", {
            "what+": "parent",
            "what-": "distant figure",
            "why+": "provided for family",
            "why-": "never emotionally available / worked constantly / criticized me",
            "when-": "childhood / rarely present",
        }, {"what+": 0.5, "what-": 0.7, "why+": 0.4, "why-": 0.8}, -0.3),
        
        # Home - mixed
        ("home", {
            "what+": "safe place",
            "what-": "place of tension",
            "why+": "mother was there",
            "why-": "parents fighting",
            "when0": "childhood"
        }, {"what+": 0.6, "what-": 0.5, "why+": 0.7, "why-": 0.6}, 0.1),
        
        # Books - escape and growth
        ("books", {
            "what+": "escape / comfort / knowledge",
            "why+": "helped me understand the world / made me feel less alone",
            "when+": "childhood / adolescence / still now",
            "where+": "bedroom / library"
        }, {"what+": 0.85, "why+": 0.9}, 0.8),
        
        # Music - emotional outlet
        ("music", {
            "what+": "emotional expression / comfort",
            "why+": "helps me process feelings / connects me to others",
            "when+": "always"
        }, {"what+": 0.8, "why+": 0.85}, 0.75),
        
        # ===== CHILDHOOD TRAUMA (Age 12) =====
        
        # Divorce
        ("divorce", {
            "what-": "family breaking apart",
            "why-": "shattered my sense of safety / changed everything",
            "when-": "age 12"
        }, {"what-": 0.9, "why-": 0.95}, -0.9),
        
        # ===== ADOLESCENCE =====
        
        # Anxiety emerges
        ("anxiety", {
            "what-": "constant companion / enemy",
            "why-": "makes everything harder / exhausting / overwhelming",
            "when-": "since adolescence / still present",
            "how-": "racing thoughts / chest tightness / avoidance"
        }, {"what-": 0.9, "why-": 0.85, "how-": 0.8}, -0.8),
        
        # School - mixed
        ("school", {
            "what+": "place of learning",
            "what-": "place of social anxiety",
            "why+": "loved learning / good grades",
            "why-": "felt like outsider / bullied sometimes"
        }, {"what+": 0.6, "what-": 0.7, "why+": 0.7, "why-": 0.75}, -0.1),
        
        # Friendship
        ("friends", {
            "what+": "connection / support",
            "what-": "source of anxiety",
            "why+": "feel understood / less alone",
            "why-": "fear of rejection / hard to trust"
        }, {"what+": 0.65, "what-": 0.6, "why+": 0.7, "why-": 0.65}, 0.2),
        
        # ===== COLLEGE =====
        
        # Psychology - found purpose
        ("psychology", {
            "what+": "calling / passion / framework for understanding",
            "why+": "explains human behavior / helps me understand myself / path to helping others"
        }, {"what+": 0.9, "why+": 0.9}, 0.85),
        
        # College
        ("college", {
            "what+": "growth / independence / discovery",
            "why+": "found my path / made real friends / became myself",
            "when+": "early twenties",
            "where+": "away from home"
        }, {"what+": 0.8, "why+": 0.85}, 0.7),
        
        # ===== SIGNIFICANT RELATIONSHIP =====
        
        # Sarah - ex partner
        ("sarah", {
            "what+": "first love",
            "what-": "painful memory",
            "why+": "showed me I could be loved / understood me",
            "why-": "left me / betrayed my trust / confirmed my fears",
            "when+": "college / three years",
            "when-": "ended badly"
        }, {"what+": 0.7, "what-": 0.85, "why+": 0.75, "why-": 0.9}, -0.3),
        
        # Love - complicated
        ("love", {
            "what+": "beautiful / meaningful",
            "what-": "dangerous / painful",
            "why+": "worth the risk / connection",
            "why-": "leads to loss / vulnerability is scary"
        }, {"what+": 0.6, "what-": 0.75, "why+": 0.65, "why-": 0.8}, 0.0),
        
        # ===== ADULTHOOD / PRESENT =====
        
        # Work as therapist
        ("work", {
            "what+": "purpose / meaning / helping others",
            "what-": "draining / heavy",
            "why+": "making a difference / using my pain to help",
            "why-": "absorb others pain / imposter syndrome",
            "when0": "daily",
            "where0": "office"
        }, {"what+": 0.8, "what-": 0.6, "why+": 0.85, "why-": 0.65}, 0.4),
        
        # Therapy (receiving)
        ("therapy", {
            "what+": "healing / growth / self-understanding",
            "why+": "processing trauma / becoming healthier"
        }, {"what+": 0.85, "why+": 0.8}, 0.7),
        
        # Meditation
        ("meditation", {
            "what+": "practice / peace / tool",
            "why+": "calms anxiety / presence / self-compassion",
            "when+": "daily morning",
            "how+": "breathing / sitting / accepting"
        }, {"what+": 0.75, "why+": 0.8}, 0.7),
        
        # Luna (dog)
        ("luna", {
            "what+": "companion / family / unconditional love",
            "why+": "always happy to see me / simple joy / teaches presence",
            "when+": "three years now",
            "where+": "home with me"
        }, {"what+": 0.95, "why+": 0.95}, 0.9),
        
        # Success - complicated
        ("success", {
            "what+": "achievement / recognition",
            "what-": "pressure / never enough",
            "why+": "proves my worth",
            "why-": "father's voice / imposter syndrome / fear of failure"
        }, {"what+": 0.5, "what-": 0.8, "why+": 0.55, "why-": 0.85}, -0.2),
        
        # Failure
        ("failure", {
            "what-": "proof I'm not enough",
            "why-": "confirms worst fears / father was right"
        }, {"what-": 0.9, "why-": 0.9}, -0.85),
        
        # Self / Me - core identity
        ("me", {
            "what+": "therapist / healer / survivor / dog mom",
            "what-": "anxious person / not enough / broken",
            "why+": "helps others / overcame trauma / capable of growth",
            "why-": "struggles with anxiety / fears abandonment / imposter"
        }, {"what+": 0.7, "what-": 0.75, "why+": 0.75, "why-": 0.8}, 0.1),
    ]
    
    # Build the buffer
    for obj_name, interrogatives, crystallizations, sentiment in experiences:
        obj_name = buffer._normalize(obj_name)
        
        # Ensure node exists
        if not buffer.has_object(obj_name):
            structure = buffer._init_valence_tagged_structure()
            buffer.graph.add_node(obj_name, sentiment=sentiment, **structure)
        
        node = buffer.graph.nodes[obj_name]
        
        # Set interrogatives and crystallizations
        for tagged_axis, content in interrogatives.items():
            if content:
                existing = node['interrogatives'].get(tagged_axis, "")
                if existing:
                    # Deduplicate
                    existing_entries = [e.strip().lower() for e in existing.split(" / ")]
                    new_entries = [e.strip() for e in content.split(" / ") 
                                  if e.strip().lower() not in existing_entries]
                    if new_entries:
                        node['interrogatives'][tagged_axis] = existing + " / " + " / ".join(new_entries)
                else:
                    node['interrogatives'][tagged_axis] = content
        
        for tagged_axis, cryst in crystallizations.items():
            node['crystallization'][tagged_axis] = cryst
        
        # Update sentiment
        node['sentiment'] = (node['sentiment'] + sentiment) / 2
        
        print(f"  ✓ {obj_name}")
    
    # Build edges (relationships between objects)
    edges = [
        # Family relationships
        ("me", "mother", "+", 1.5),
        ("me", "father", "-", 1.2),
        ("mother", "home", "+", 1.0),
        ("father", "home", "-", 0.8),
        ("father", "divorce", "-", 1.5),
        ("mother", "divorce", "-", 1.0),
        ("divorce", "me", "-", 1.5),
        
        # Coping mechanisms
        ("me", "books", "+", 1.3),
        ("me", "music", "+", 1.2),
        ("anxiety", "books", "+", 0.8),
        ("anxiety", "meditation", "+", 1.0),
        
        # Mental health
        ("me", "anxiety", "-", 1.5),
        ("anxiety", "work", "-", 0.7),
        ("anxiety", "friends", "-", 0.6),
        ("me", "therapy", "+", 1.2),
        ("me", "meditation", "+", 1.0),
        
        # Career
        ("me", "psychology", "+", 1.4),
        ("psychology", "work", "+", 1.2),
        ("college", "psychology", "+", 1.0),
        ("me", "work", "+", 1.3),
        
        # Relationships
        ("me", "sarah", "-", 1.0),
        ("sarah", "love", "+", 0.8),
        ("sarah", "love", "-", 1.2),
        ("me", "love", "+", 0.7),
        ("me", "love", "-", 0.9),
        
        # Present life
        ("me", "luna", "+", 1.5),
        ("luna", "home", "+", 1.0),
        
        # Self-worth issues
        ("father", "success", "-", 1.2),
        ("father", "failure", "-", 1.3),
        ("me", "success", "-", 0.9),
        ("me", "failure", "-", 1.2),
        ("success", "work", "+", 0.8),
        
        # School/college chain
        ("me", "school", "0", 1.0),
        ("school", "college", "+", 0.8),
        ("me", "college", "+", 1.1),
        ("college", "sarah", "+", 0.9),
        ("me", "friends", "+", 0.9),
    ]
    
    print("\nCreating relationships...")
    
    for source, target, valence, weight in edges:
        source = buffer._normalize(source)
        target = buffer._normalize(target)
        
        if source in buffer.graph and target in buffer.graph:
            key = f"cooccur_{valence}"
            if not buffer.graph.has_edge(source, target, key=key):
                buffer.graph.add_edge(source, target, key=key, weight=weight, valence=valence)
    
    # Save the buffer
    buffer.save_buffer(force=True)
    
    print("\n" + "=" * 60)
    print("SEEDING COMPLETE")
    print("=" * 60)
    print(f"\nObjects created: {len(buffer.graph.nodes())}")
    print(f"Relationships created: {len(buffer.graph.edges())}")
    
    print("\n" + "-" * 60)
    print("KEY OBJECTS:")
    print("-" * 60)
    for node in sorted(buffer.graph.nodes()):
        degree = buffer.graph.degree(node)
        sentiment = buffer.graph.nodes[node].get('sentiment', 0)
        icon = "🟢" if sentiment > 0.2 else "🔴" if sentiment < -0.2 else "⚪"
        print(f"  {icon} {node}: {degree} connections, sentiment {sentiment:.2f}")
    
    print("\n" + "=" * 60)
    print("TEST INPUTS TO TRY:")
    print("=" * 60)
    print("""
POSITIVE TESTS:
  • "Luna greeted me at the door"
  • "My therapy session went well"
  • "I helped a client break through today"
  • "Mom called to check on me"

NEGATIVE TESTS:
  • "I saw my father today"
  • "I feel like a failure"
  • "Work was overwhelming"
  • "I'm not good enough"

COMPLEX/AMBIVALENT TESTS:
  • "I thought about Sarah today"
  • "Someone asked about my childhood"
  • "I received praise at work"
  • "Love feels impossible"

CASCADE TESTS (should propagate through network):
  • "My father said he's proud of me"
  • "I had a panic attack at work"
  • "Luna got sick"
""")
    
    return processor


if __name__ == "__main__":
    processor = seed_complex_history()
    print("\n✓ Buffer saved. Run 'streamlit run ui/app.py' to explore.")
